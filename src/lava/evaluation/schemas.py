"""Typed records for LAVA references, predictions, folds, and scores."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AnswerFormat(StrEnum):
    """Canonical answer formats defined by the LAVA challenge."""

    STRING = "string"
    NUMBER = "number"
    UNORDERED_LIST = "unordered_list"
    ORDERED_LIST = "ordered_list"

    @classmethod
    def from_raw(cls, value: str) -> AnswerFormat:
        """Map a source value to a canonical LAVA answer format."""
        normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
        aliases = {
            "string": cls.STRING,
            "short_text": cls.STRING,
            "text": cls.STRING,
            "number": cls.NUMBER,
            "numeric": cls.NUMBER,
            "unordered_list": cls.UNORDERED_LIST,
            "unordered": cls.UNORDERED_LIST,
            "list": cls.UNORDERED_LIST,
            "ordered_list": cls.ORDERED_LIST,
            "ordered": cls.ORDERED_LIST,
        }
        try:
            return aliases[normalized]
        except KeyError as error:
            valid = ", ".join(sorted(aliases))
            message = f"Unsupported answer format {value!r}; expected one of: {valid}"
            raise ValueError(message) from error


class FrozenModel(BaseModel):
    """Immutable strict Pydantic base model."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ReferenceRecord(FrozenModel):
    """One labeled LAVA question and its evidence pages."""

    question_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer_format: AnswerFormat
    answer: str = Field(min_length=1)
    evidence_pages: tuple[int, ...] = Field(min_length=1)
    language: str = Field(min_length=2, max_length=8)

    @field_validator("evidence_pages")
    @classmethod
    def validate_evidence_pages(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Require positive, unique, sorted one-indexed PDF pages."""
        if any(page < 1 for page in value):
            message = "Evidence pages must be positive one-indexed integers"
            raise ValueError(message)
        normalized = tuple(sorted(set(value)))
        if normalized != value:
            message = "Evidence pages must already be unique and sorted"
            raise ValueError(message)
        return value


class PredictionRecord(FrozenModel):
    """One model answer and its predicted evidence pages."""

    question_id: str = Field(min_length=1)
    answer: str
    evidence_pages: tuple[int, ...]

    @field_validator("evidence_pages")
    @classmethod
    def validate_evidence_pages(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Require nonnegative cardinality and positive unique pages."""
        if any(page < 1 for page in value):
            message = "Predicted evidence pages must be positive one-indexed integers"
            raise ValueError(message)
        normalized = tuple(sorted(set(value)))
        if normalized != value:
            message = "Predicted evidence pages must already be unique and sorted"
            raise ValueError(message)
        return value


class FoldManifest(FrozenModel):
    """One leave-one-document-out fold."""

    fold_id: str = Field(min_length=1)
    validation_document_id: str = Field(min_length=1)
    training_document_ids: tuple[str, ...] = Field(min_length=1)
    training_question_ids: tuple[str, ...] = Field(min_length=1)
    validation_question_ids: tuple[str, ...] = Field(min_length=1)


class QuestionScore(FrozenModel):
    """Per-question answer, grounding, and combined scores."""

    question_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    language: str = Field(min_length=2, max_length=8)
    answer_format: AnswerFormat
    answer_score: float = Field(ge=0.0, le=1.0)
    grounding_score: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)


class InnerFoldManifest(FrozenModel):
    """One inner document holdout inside an outer training partition."""

    fold_id: str = Field(min_length=1)
    validation_document_id: str = Field(min_length=1)
    training_document_ids: tuple[str, ...] = Field(min_length=1)
    training_question_ids: tuple[str, ...] = Field(min_length=1)
    validation_question_ids: tuple[str, ...] = Field(min_length=1)


class NestedFoldManifest(FrozenModel):
    """One outer document holdout with nested inner document holdouts."""

    fold_id: str = Field(min_length=1)
    validation_document_id: str = Field(min_length=1)
    training_document_ids: tuple[str, ...] = Field(min_length=1)
    training_question_ids: tuple[str, ...] = Field(min_length=1)
    validation_question_ids: tuple[str, ...] = Field(min_length=1)
    inner_folds: tuple[InnerFoldManifest, ...] = Field(min_length=1)
