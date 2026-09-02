"""Official-spec LAVA answer and evidence-grounding metric engine."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from statistics import fmean
from typing import Any

from lava.evaluation.judges import SemanticJudge
from lava.evaluation.matching import (
    maximum_bipartite_matches,
    semantic_lcs_length,
    semantic_match_matrix,
)
from lava.evaluation.normalization import parse_list_answer
from lava.evaluation.schemas import (
    AnswerFormat,
    PredictionRecord,
    QuestionScore,
    ReferenceRecord,
)


def set_f1(reference: Sequence[int], prediction: Sequence[int]) -> float:
    """Compute exact set F1, including well-defined empty-set behavior."""
    reference_set = set(reference)
    prediction_set = set(prediction)
    if not reference_set and not prediction_set:
        return 1.0
    if not reference_set or not prediction_set:
        return 0.0
    matches = len(reference_set & prediction_set)
    return 2.0 * matches / (len(reference_set) + len(prediction_set))


def _unordered_list_score(
    reference: str,
    prediction: str,
    *,
    judge: SemanticJudge,
    language: str,
) -> float:
    reference_items = parse_list_answer(reference)
    prediction_items = parse_list_answer(prediction)
    if not reference_items and not prediction_items:
        return 1.0
    if not reference_items or not prediction_items:
        return 0.0
    matrix = semantic_match_matrix(
        reference_items,
        prediction_items,
        judge=judge,
        language=language,
    )
    matches = maximum_bipartite_matches(matrix)
    return 2.0 * matches / (len(reference_items) + len(prediction_items))


def _ordered_list_score(
    reference: str,
    prediction: str,
    *,
    judge: SemanticJudge,
    language: str,
) -> float:
    reference_items = parse_list_answer(reference)
    prediction_items = parse_list_answer(prediction)
    denominator = max(len(reference_items), len(prediction_items))
    if denominator == 0:
        return 1.0
    length = semantic_lcs_length(
        reference_items,
        prediction_items,
        judge=judge,
        language=language,
    )
    return length / denominator


def answer_score(
    reference: ReferenceRecord,
    prediction: PredictionRecord,
    *,
    judge: SemanticJudge,
) -> float:
    """Score one answer according to its declared LAVA answer format."""
    if reference.answer_format in {AnswerFormat.STRING, AnswerFormat.NUMBER}:
        return float(
            judge.equivalent(
                reference.answer,
                prediction.answer,
                language=reference.language,
            )
        )
    if reference.answer_format is AnswerFormat.UNORDERED_LIST:
        return _unordered_list_score(
            reference.answer,
            prediction.answer,
            judge=judge,
            language=reference.language,
        )
    return _ordered_list_score(
        reference.answer,
        prediction.answer,
        judge=judge,
        language=reference.language,
    )


def score_question(
    reference: ReferenceRecord,
    prediction: PredictionRecord,
    *,
    judge: SemanticJudge,
) -> QuestionScore:
    """Compute answer, grounding, and combined score for one question."""
    if prediction.question_id != reference.question_id:
        message = "Prediction and reference question IDs do not match"
        raise ValueError(message)
    answer = answer_score(reference, prediction, judge=judge)
    grounding = set_f1(reference.evidence_pages, prediction.evidence_pages)
    return QuestionScore(
        question_id=reference.question_id,
        document_id=reference.document_id,
        language=reference.language,
        answer_format=reference.answer_format,
        answer_score=answer,
        grounding_score=grounding,
        overall_score=(answer + grounding) / 2.0,
    )


def _mean_scores(scores: Sequence[QuestionScore]) -> dict[str, float]:
    if not scores:
        return {"answer": 0.0, "grounding": 0.0, "overall": 0.0}
    return {
        "answer": fmean(score.answer_score for score in scores),
        "grounding": fmean(score.grounding_score for score in scores),
        "overall": fmean(score.overall_score for score in scores),
    }


def score_dataset(
    references: Iterable[ReferenceRecord],
    predictions: Iterable[PredictionRecord],
    *,
    judge: SemanticJudge,
) -> tuple[tuple[QuestionScore, ...], dict[str, Any]]:
    """Score a complete prediction set and return micro, macro, and slice metrics."""
    reference_records = tuple(references)
    prediction_records = tuple(predictions)
    reference_by_id = {record.question_id: record for record in reference_records}
    prediction_by_id = {record.question_id: record for record in prediction_records}
    if len(reference_by_id) != len(reference_records):
        message = "Reference question IDs must be unique"
        raise ValueError(message)
    if len(prediction_by_id) != len(prediction_records):
        message = "Prediction question IDs must be unique"
        raise ValueError(message)
    missing = sorted(set(reference_by_id) - set(prediction_by_id))
    extra = sorted(set(prediction_by_id) - set(reference_by_id))
    if missing or extra:
        message = f"Prediction coverage mismatch; missing={missing}, extra={extra}"
        raise ValueError(message)

    scores = tuple(
        score_question(reference_by_id[question_id], prediction_by_id[question_id], judge=judge)
        for question_id in sorted(reference_by_id)
    )
    by_document: dict[str, list[QuestionScore]] = defaultdict(list)
    by_language: dict[str, list[QuestionScore]] = defaultdict(list)
    by_format: dict[str, list[QuestionScore]] = defaultdict(list)
    for score in scores:
        by_document[score.document_id].append(score)
        by_language[score.language].append(score)
        by_format[score.answer_format.value].append(score)

    document_means = [_mean_scores(group) for group in by_document.values()]
    document_macro = {
        key: fmean(result[key] for result in document_means)
        for key in ("answer", "grounding", "overall")
    }
    summary: dict[str, Any] = {
        "question_count": len(scores),
        "document_count": len(by_document),
        "question_micro": _mean_scores(scores),
        "document_macro": document_macro,
        "by_document": {key: _mean_scores(value) for key, value in sorted(by_document.items())},
        "by_language": {key: _mean_scores(value) for key, value in sorted(by_language.items())},
        "by_answer_format": {key: _mean_scores(value) for key, value in sorted(by_format.items())},
        "answer_format_counts": dict(
            sorted(Counter(score.answer_format.value for score in scores).items())
        ),
    }
    return scores, summary
