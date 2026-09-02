"""Evidence-page retrieval metrics for multilingual multi-page Document VQA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True, slots=True)
class RetrievalScore:
    """Per-question retrieval metrics at one page budget."""

    k: int
    recall_at_k: float
    all_evidence_at_k: float
    reciprocal_rank_at_k: float
    average_precision_at_k: float
    ndcg_at_k: float


def _validate_inputs(
    ranked_pages: tuple[int, ...],
    relevant_pages: frozenset[int],
    k: int,
) -> None:
    if k < 1:
        message = "k must be at least one"
        raise ValueError(message)
    if not relevant_pages:
        message = "relevant_pages must be nonempty"
        raise ValueError(message)
    if len(ranked_pages) != len(set(ranked_pages)):
        message = "ranked_pages must not contain duplicates"
        raise ValueError(message)
    if any(page < 1 for page in (*ranked_pages, *relevant_pages)):
        message = "Page numbers must be positive and one-indexed"
        raise ValueError(message)


def score_retrieval(
    ranked_pages: tuple[int, ...],
    relevant_pages: frozenset[int],
    *,
    k: int,
) -> RetrievalScore:
    """Measure evidence coverage and ranking quality at a fixed page budget."""
    _validate_inputs(ranked_pages, relevant_pages, k)
    top_k = ranked_pages[:k]
    hits = tuple(int(page in relevant_pages) for page in top_k)
    hit_count = sum(hits)
    recall = hit_count / len(relevant_pages)
    all_evidence = float(relevant_pages.issubset(top_k))
    first_hit = next((rank for rank, hit in enumerate(hits, start=1) if hit), None)
    reciprocal_rank = 0.0 if first_hit is None else 1.0 / first_hit
    precision_sum = 0.0
    cumulative_hits = 0
    for rank, hit in enumerate(hits, start=1):
        cumulative_hits += hit
        if hit:
            precision_sum += cumulative_hits / rank
    average_precision = precision_sum / min(len(relevant_pages), k)
    discounted_gain = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
    ideal_hit_count = min(len(relevant_pages), k)
    ideal_gain = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hit_count + 1))
    ndcg = discounted_gain / ideal_gain if ideal_gain else 0.0
    return RetrievalScore(
        k=k,
        recall_at_k=recall,
        all_evidence_at_k=all_evidence,
        reciprocal_rank_at_k=reciprocal_rank,
        average_precision_at_k=average_precision,
        ndcg_at_k=ndcg,
    )


def score_retrieval_budgets(
    ranked_pages: tuple[int, ...],
    relevant_pages: frozenset[int],
    *,
    budgets: tuple[int, ...] = (1, 2, 3, 5, 10),
) -> tuple[RetrievalScore, ...]:
    """Score a retrieval ranking at prespecified page budgets."""
    if (
        not budgets
        or any(budget < 1 for budget in budgets)
        or (tuple(sorted(set(budgets))) != budgets)
    ):
        message = "budgets must be positive, nonempty, unique, and sorted"
        raise ValueError(message)
    return tuple(score_retrieval(ranked_pages, relevant_pages, k=budget) for budget in budgets)


def aggregate_retrieval_scores(
    scores: tuple[RetrievalScore, ...],
) -> dict[int, dict[str, float]]:
    """Aggregate per-question retrieval scores by page budget."""
    if not scores:
        message = "At least one retrieval score is required"
        raise ValueError(message)
    grouped: dict[int, list[RetrievalScore]] = {}
    for score in scores:
        grouped.setdefault(score.k, []).append(score)
    return {
        k: {
            "recall_at_k": fmean(item.recall_at_k for item in group),
            "all_evidence_at_k": fmean(item.all_evidence_at_k for item in group),
            "reciprocal_rank_at_k": fmean(item.reciprocal_rank_at_k for item in group),
            "average_precision_at_k": fmean(item.average_precision_at_k for item in group),
            "ndcg_at_k": fmean(item.ndcg_at_k for item in group),
        }
        for k, group in sorted(grouped.items())
    }
