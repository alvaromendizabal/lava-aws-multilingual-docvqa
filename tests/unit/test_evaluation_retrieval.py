"""Tests for evidence-page retrieval metrics."""

import pytest

from lava.evaluation.retrieval import (
    aggregate_retrieval_scores,
    score_retrieval,
    score_retrieval_budgets,
)


def test_retrieval_metrics_measure_partial_and_complete_evidence() -> None:
    """Recall and all-evidence success must distinguish partial from complete coverage."""
    score = score_retrieval((3, 1, 2, 4), frozenset({1, 2}), k=2)
    assert score.recall_at_k == 0.5
    assert score.all_evidence_at_k == 0.0
    assert score.reciprocal_rank_at_k == 0.5


def test_retrieval_metrics_reward_early_relevant_pages() -> None:
    """Ranking metrics should prefer evidence that appears earlier."""
    early = score_retrieval((1, 3, 2), frozenset({1, 2}), k=3)
    late = score_retrieval((3, 1, 2), frozenset({1, 2}), k=3)
    assert early.average_precision_at_k > late.average_precision_at_k
    assert early.ndcg_at_k > late.ndcg_at_k


def test_retrieval_budgets_and_aggregation_are_deterministic() -> None:
    """Prespecified budgets must be sorted, unique, and aggregatable."""
    scores = score_retrieval_budgets(
        (1, 2, 3),
        frozenset({2}),
        budgets=(1, 2, 3),
    )
    aggregated = aggregate_retrieval_scores(scores)
    assert tuple(aggregated) == (1, 2, 3)
    assert aggregated[1]["recall_at_k"] == 0.0
    assert aggregated[2]["recall_at_k"] == 1.0
    with pytest.raises(ValueError, match="sorted"):
        score_retrieval_budgets((1, 2), frozenset({1}), budgets=(2, 1))


def test_duplicate_ranked_pages_fail_closed() -> None:
    """A ranking cannot contain the same page more than once."""
    with pytest.raises(ValueError, match="duplicates"):
        score_retrieval((1, 1), frozenset({1}), k=2)
