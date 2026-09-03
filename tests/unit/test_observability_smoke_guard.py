"""Tests for fail-closed first-smoke plan validation."""

from __future__ import annotations

import pytest

from lava.observability.smoke_guard import validate_first_smoke_plan


def _safe_plan() -> dict[str, object]:
    return {
        "model_key": "qwen35_4b_fused_direct",
        "instance_type": "ml.g6e.2xlarge",
        "limit": 1,
        "instance_count": 1,
        "max_runtime_seconds": 3600,
        "creates_endpoint": False,
        "managed_spot": False,
    }


def test_safe_first_smoke_plan_passes() -> None:
    """The exact bounded smoke plan must pass."""
    validate_first_smoke_plan(_safe_plan())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("limit", 2),
        ("instance_count", 2),
        ("max_runtime_seconds", 3601),
        ("creates_endpoint", True),
        ("managed_spot", True),
        ("model_key", "qwen35_9b_fused_direct"),
    ],
)
def test_unsafe_first_smoke_plan_fails(field: str, value: object) -> None:
    """Any expansion of the first paid smoke job must fail closed."""
    plan = _safe_plan()
    plan[field] = value
    with pytest.raises(RuntimeError, match="Unsafe first-smoke plan"):
        validate_first_smoke_plan(plan)
