"""Tests for answer, grounding, and aggregate LAVA scoring."""

from dataclasses import dataclass

import pytest

from lava.evaluation.judges import CachedSemanticJudge, NormalizedExactJudge
from lava.evaluation.matching import maximum_bipartite_matches
from lava.evaluation.metric import score_dataset, score_question, set_f1
from lava.evaluation.schemas import AnswerFormat, PredictionRecord, ReferenceRecord


@dataclass(frozen=True, slots=True)
class MappingJudge:
    """Deterministic relation-backed judge for matching tests."""

    pairs: frozenset[tuple[str, str]]
    identity: str = "mapping-judge-v1"

    def equivalent(self, reference: str, prediction: str, *, language: str) -> bool:
        """Return whether the configured relation contains the pair."""
        del language
        return (reference, prediction) in self.pairs


def _reference(
    *,
    question_id: str = "q1",
    document_id: str = "d1",
    answer_format: AnswerFormat = AnswerFormat.STRING,
    answer: str = "東京",
    pages: tuple[int, ...] = (1,),
) -> ReferenceRecord:
    return ReferenceRecord(
        question_id=question_id,
        document_id=document_id,
        question="質問",
        answer_format=answer_format,
        answer=answer,
        evidence_pages=pages,
        language="ja",
    )


def test_grounding_set_f1() -> None:
    """Grounding follows exact set F1."""
    assert set_f1((1, 2), (2, 3)) == pytest.approx(0.5)


def test_scalar_and_grounding_combination() -> None:
    """Question score is the arithmetic mean of answer and grounding."""
    score = score_question(
        _reference(pages=(1, 2)),
        PredictionRecord(question_id="q1", answer="東京", evidence_pages=(2,)),
        judge=NormalizedExactJudge(),
    )
    assert score.answer_score == 1.0
    assert score.grounding_score == pytest.approx(2.0 / 3.0)
    assert score.overall_score == pytest.approx(5.0 / 6.0)


def test_unordered_list_uses_global_one_to_one_matching() -> None:
    """Maximum matching avoids the order dependence of greedy item assignment."""
    judge = MappingJudge(
        frozenset(
            {
                ("a", "x"),
                ("a", "y"),
                ("b", "x"),
            }
        )
    )
    score = score_question(
        _reference(answer_format=AnswerFormat.UNORDERED_LIST, answer='["a", "b"]'),
        PredictionRecord(question_id="q1", answer='["x", "y"]', evidence_pages=(1,)),
        judge=judge,
    )
    assert score.answer_score == 1.0


def test_ordered_list_uses_semantic_lcs() -> None:
    """Ordered answers should preserve sequence order."""
    score = score_question(
        _reference(answer_format=AnswerFormat.ORDERED_LIST, answer='["a", "b", "c"]'),
        PredictionRecord(question_id="q1", answer='["a", "c", "b"]', evidence_pages=(1,)),
        judge=NormalizedExactJudge(),
    )
    assert score.answer_score == pytest.approx(2.0 / 3.0)


def test_bipartite_matrix_must_be_rectangular() -> None:
    """Malformed match matrices should fail rather than silently mis-score."""
    with pytest.raises(ValueError, match="equal length"):
        maximum_bipartite_matches(((True,), (True, False)))


def test_dataset_requires_complete_prediction_coverage() -> None:
    """Missing predictions must fail closed."""
    with pytest.raises(ValueError, match="coverage mismatch"):
        score_dataset((_reference(),), (), judge=NormalizedExactJudge())


def test_cached_judge_reuses_pair_decisions() -> None:
    """Repeated semantic comparisons should be memoized."""
    judge = CachedSemanticJudge(NormalizedExactJudge())
    assert judge.equivalent("1,000", "1000", language="ja")
    assert judge.equivalent("1,000", "1000", language="ja")
    assert judge.cache_size == 1
