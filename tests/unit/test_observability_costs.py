"""Tests for conservative operator-supplied cost limits."""

from __future__ import annotations

import pytest

from lava.observability.costs import enforce_cost_cap, estimate_maximum_cost


def test_estimate_uses_runtime_instances_and_contingency() -> None:
    """The estimate must include all explicit bound components."""
    estimate = estimate_maximum_cost(
        hourly_usd_ceiling=10.0,
        max_runtime_seconds=3600,
        instance_count=1,
        contingency_factor=1.25,
    )
    assert estimate.estimated_maximum_usd == 12.5


def test_cost_guard_fails_closed_above_cap() -> None:
    """A plan above the operator cap must be rejected."""
    estimate = estimate_maximum_cost(
        hourly_usd_ceiling=10.0,
        max_runtime_seconds=3600,
        instance_count=1,
        contingency_factor=1.25,
    )
    with pytest.raises(RuntimeError, match="exceeds the allowed cap"):
        enforce_cost_cap(estimate, maximum_allowed_usd=12.49)


def test_cost_guard_accepts_equal_cap() -> None:
    """An estimate equal to the explicit cap is acceptable."""
    estimate = estimate_maximum_cost(
        hourly_usd_ceiling=10.0,
        max_runtime_seconds=3600,
        instance_count=1,
        contingency_factor=1.25,
    )
    enforce_cost_cap(estimate, maximum_allowed_usd=12.5)
