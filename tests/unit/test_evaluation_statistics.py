"""Tests for exact paired comparison utilities."""

import pytest

from lava.evaluation.statistics import (
    cluster_bootstrap_mean_interval,
    compare_document_scores,
    exact_paired_sign_flip_p_value,
)


def test_exact_sign_flip_is_bounded() -> None:
    """Exact randomization p-values must remain in the unit interval."""
    value = exact_paired_sign_flip_p_value((0.1, 0.2, 0.3, 0.4, 0.5))
    assert 0.0 <= value <= 1.0


def test_bootstrap_is_deterministic_for_seed() -> None:
    """The same seed and values should reproduce the same interval."""
    first = cluster_bootstrap_mean_interval((0.1, 0.2, 0.3), seed=42, resamples=100)
    second = cluster_bootstrap_mean_interval((0.1, 0.2, 0.3), seed=42, resamples=100)
    assert first == second


def test_document_comparison_reports_paired_effects() -> None:
    """Comparison output should retain every paired document delta."""
    result = compare_document_scores(
        {"a": 0.5, "b": 0.6},
        {"a": 0.7, "b": 0.5},
        seed=7,
    )
    assert result["document_count"] == 2
    assert result["documents_improved"] == 1
    assert result["mean_delta"] == pytest.approx(0.05)
