"""Small-sample paired uncertainty tools for document-level model comparisons."""

from __future__ import annotations

import itertools
import random
from collections.abc import Mapping, Sequence
from statistics import fmean, median
from typing import Any


def exact_paired_sign_flip_p_value(deltas: Sequence[float]) -> float:
    """Compute an exact two-sided paired sign-flip p-value across documents."""
    if not deltas:
        message = "At least one paired document delta is required"
        raise ValueError(message)
    observed = abs(fmean(deltas))
    outcomes = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(deltas)):
        outcomes.append(abs(fmean(sign * delta for sign, delta in zip(signs, deltas, strict=True))))
    return sum(outcome >= observed - 1e-15 for outcome in outcomes) / len(outcomes)


def cluster_bootstrap_mean_interval(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a deterministic percentile interval from document-cluster resampling."""
    if not values:
        message = "At least one document score is required"
        raise ValueError(message)
    if resamples < 1:
        message = "resamples must be positive"
        raise ValueError(message)
    if not 0.0 < confidence < 1.0:
        message = "confidence must lie strictly between zero and one"
        raise ValueError(message)
    generator = random.Random(seed)
    sample_size = len(values)
    means = sorted(
        fmean(values[generator.randrange(sample_size)] for _ in range(sample_size))
        for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0
    lower_index = min(int(tail * resamples), resamples - 1)
    upper_index = min(int((1.0 - tail) * resamples), resamples - 1)
    return means[lower_index], means[upper_index]


def compare_document_scores(
    champion: Mapping[str, float],
    challenger: Mapping[str, float],
    *,
    seed: int,
) -> dict[str, Any]:
    """Compare paired document scores without pretending five documents are a large sample."""
    if set(champion) != set(challenger):
        message = "Champion and challenger must cover the same documents"
        raise ValueError(message)
    documents = sorted(champion)
    deltas = [challenger[document] - champion[document] for document in documents]
    lower, upper = cluster_bootstrap_mean_interval(deltas, seed=seed)
    return {
        "document_count": len(documents),
        "mean_delta": fmean(deltas),
        "median_delta": median(deltas),
        "minimum_delta": min(deltas),
        "maximum_delta": max(deltas),
        "documents_improved": sum(delta > 0.0 for delta in deltas),
        "documents_tied": sum(delta == 0.0 for delta in deltas),
        "documents_regressed": sum(delta < 0.0 for delta in deltas),
        "exact_two_sided_sign_flip_p_value": exact_paired_sign_flip_p_value(deltas),
        "exploratory_cluster_bootstrap_95_interval": [lower, upper],
        "per_document_delta": dict(zip(documents, deltas, strict=True)),
    }
