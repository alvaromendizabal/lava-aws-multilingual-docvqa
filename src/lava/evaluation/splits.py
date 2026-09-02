"""Nested document-isolated split construction for the scarce LAVA labels."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from lava.evaluation.schemas import (
    FoldManifest,
    InnerFoldManifest,
    NestedFoldManifest,
    ReferenceRecord,
)


def _records_by_document(
    records: Iterable[ReferenceRecord],
) -> tuple[tuple[ReferenceRecord, ...], dict[str, list[ReferenceRecord]]]:
    materialized = tuple(records)
    if not materialized:
        message = "At least one reference record is required"
        raise ValueError(message)
    records_by_document: dict[str, list[ReferenceRecord]] = defaultdict(list)
    all_question_ids: set[str] = set()
    for record in materialized:
        if record.question_id in all_question_ids:
            message = f"Duplicate question ID: {record.question_id}"
            raise ValueError(message)
        all_question_ids.add(record.question_id)
        records_by_document[record.document_id].append(record)
    return materialized, records_by_document


def build_leave_one_document_out_folds(
    records: Iterable[ReferenceRecord],
) -> tuple[FoldManifest, ...]:
    """Build one deterministic validation fold per source document."""
    materialized, records_by_document = _records_by_document(records)
    document_ids = tuple(sorted(records_by_document))
    if len(document_ids) < 2:
        message = "Leave-one-document-out evaluation requires at least two documents"
        raise ValueError(message)
    folds: list[FoldManifest] = []
    for fold_number, validation_document in enumerate(document_ids, start=1):
        training_documents = tuple(
            document for document in document_ids if document != validation_document
        )
        validation_questions = tuple(
            sorted(record.question_id for record in records_by_document[validation_document])
        )
        training_questions = tuple(
            sorted(
                record.question_id
                for document in training_documents
                for record in records_by_document[document]
            )
        )
        folds.append(
            FoldManifest(
                fold_id=f"outer-{fold_number:02d}",
                validation_document_id=validation_document,
                training_document_ids=training_documents,
                training_question_ids=training_questions,
                validation_question_ids=validation_questions,
            )
        )
    validate_folds(folds, materialized)
    return tuple(folds)


def build_nested_leave_one_document_out_folds(
    records: Iterable[ReferenceRecord],
) -> tuple[NestedFoldManifest, ...]:
    """Build outer LODO folds and inner LODO folds within each outer training set."""
    materialized, records_by_document = _records_by_document(records)
    document_ids = tuple(sorted(records_by_document))
    if len(document_ids) < 3:
        message = "Nested document holdout requires at least three documents"
        raise ValueError(message)
    outer_folds = build_leave_one_document_out_folds(materialized)
    nested_folds: list[NestedFoldManifest] = []
    for outer in outer_folds:
        inner_folds: list[InnerFoldManifest] = []
        for inner_number, inner_validation_document in enumerate(
            outer.training_document_ids,
            start=1,
        ):
            inner_training_documents = tuple(
                document
                for document in outer.training_document_ids
                if document != inner_validation_document
            )
            inner_training_questions = tuple(
                sorted(
                    record.question_id
                    for document in inner_training_documents
                    for record in records_by_document[document]
                )
            )
            inner_validation_questions = tuple(
                sorted(
                    record.question_id for record in records_by_document[inner_validation_document]
                )
            )
            inner_folds.append(
                InnerFoldManifest(
                    fold_id=f"{outer.fold_id}-inner-{inner_number:02d}",
                    validation_document_id=inner_validation_document,
                    training_document_ids=inner_training_documents,
                    training_question_ids=inner_training_questions,
                    validation_question_ids=inner_validation_questions,
                )
            )
        nested_folds.append(
            NestedFoldManifest(
                fold_id=outer.fold_id,
                validation_document_id=outer.validation_document_id,
                training_document_ids=outer.training_document_ids,
                training_question_ids=outer.training_question_ids,
                validation_question_ids=outer.validation_question_ids,
                inner_folds=tuple(inner_folds),
            )
        )
    validate_nested_folds(nested_folds, materialized)
    return tuple(nested_folds)


def validate_folds(
    folds: Iterable[FoldManifest],
    records: Iterable[ReferenceRecord],
) -> None:
    """Assert complete validation coverage and strict outer isolation."""
    fold_list = tuple(folds)
    record_list = tuple(records)
    document_ids = {record.document_id for record in record_list}
    question_ids = {record.question_id for record in record_list}
    validation_documents: list[str] = []
    validation_questions: list[str] = []
    for fold in fold_list:
        training_document_set = set(fold.training_document_ids)
        training_question_set = set(fold.training_question_ids)
        validation_question_set = set(fold.validation_question_ids)
        if fold.validation_document_id in training_document_set:
            message = f"Document crossover in {fold.fold_id}"
            raise ValueError(message)
        if training_question_set & validation_question_set:
            message = f"Question crossover in {fold.fold_id}"
            raise ValueError(message)
        if training_question_set | validation_question_set != question_ids:
            message = f"Incomplete question assignment in {fold.fold_id}"
            raise ValueError(message)
        validation_documents.append(fold.validation_document_id)
        validation_questions.extend(fold.validation_question_ids)
    if set(validation_documents) != document_ids or len(validation_documents) != len(document_ids):
        message = "Every document must be held out exactly once"
        raise ValueError(message)
    if set(validation_questions) != question_ids or len(validation_questions) != len(question_ids):
        message = "Every question must be validated exactly once"
        raise ValueError(message)


def validate_nested_folds(
    folds: Iterable[NestedFoldManifest],
    records: Iterable[ReferenceRecord],
) -> None:
    """Reject outer or inner document leakage and malformed nested coverage."""
    fold_list = tuple(folds)
    record_list = tuple(records)
    outer_folds = tuple(
        FoldManifest(
            fold_id=fold.fold_id,
            validation_document_id=fold.validation_document_id,
            training_document_ids=fold.training_document_ids,
            training_question_ids=fold.training_question_ids,
            validation_question_ids=fold.validation_question_ids,
        )
        for fold in fold_list
    )
    validate_folds(outer_folds, record_list)
    record_by_question = {record.question_id: record for record in record_list}
    for outer in fold_list:
        outer_training_documents = set(outer.training_document_ids)
        outer_training_questions = set(outer.training_question_ids)
        inner_validation_documents = [inner.validation_document_id for inner in outer.inner_folds]
        inner_validation_questions: list[str] = []
        if set(inner_validation_documents) != outer_training_documents or len(
            inner_validation_documents
        ) != len(outer_training_documents):
            message = f"Malformed inner document coverage in {outer.fold_id}"
            raise ValueError(message)
        for inner in outer.inner_folds:
            inner_training_documents = set(inner.training_document_ids)
            if outer.validation_document_id in inner_training_documents:
                message = f"Outer validation leakage in {inner.fold_id}"
                raise ValueError(message)
            if inner.validation_document_id in inner_training_documents:
                message = f"Inner validation leakage in {inner.fold_id}"
                raise ValueError(message)
            if inner_training_documents | {inner.validation_document_id} != (
                outer_training_documents
            ):
                message = f"Incomplete inner document assignment in {inner.fold_id}"
                raise ValueError(message)
            inner_training_questions = set(inner.training_question_ids)
            inner_validation_question_set = set(inner.validation_question_ids)
            if inner_training_questions & inner_validation_question_set:
                message = f"Inner question crossover in {inner.fold_id}"
                raise ValueError(message)
            if inner_training_questions | inner_validation_question_set != (
                outer_training_questions
            ):
                message = f"Incomplete inner question assignment in {inner.fold_id}"
                raise ValueError(message)
            if any(
                record_by_question[question_id].document_id == outer.validation_document_id
                for question_id in inner_training_questions | inner_validation_question_set
            ):
                message = f"Outer validation questions entered {inner.fold_id}"
                raise ValueError(message)
            inner_validation_questions.extend(inner.validation_question_ids)
        if set(inner_validation_questions) != outer_training_questions or len(
            inner_validation_questions
        ) != len(outer_training_questions):
            message = (
                f"Every outer-training question must be inner validation once in {outer.fold_id}"
            )
            raise ValueError(message)
