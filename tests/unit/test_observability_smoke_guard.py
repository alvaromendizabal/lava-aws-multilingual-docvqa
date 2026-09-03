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


class _FakePaginator:
    def __init__(self, quotas: list[dict[str, object]]) -> None:
        self.quotas = quotas

    def paginate(self, **_: object) -> list[dict[str, object]]:
        return [{"Quotas": self.quotas}]


class _FakeServiceQuotas:
    def __init__(self, quotas: list[dict[str, object]]) -> None:
        self.quotas = quotas

    def get_paginator(self, name: str) -> _FakePaginator:
        assert name == "list_service_quotas"
        return _FakePaginator(self.quotas)


def test_exact_on_demand_quota_is_selected() -> None:
    """Spot quota appearing first must never satisfy an on-demand plan."""
    from lava.observability.smoke_guard import verify_training_quota

    client = _FakeServiceQuotas(
        [
            {
                "QuotaName": "ml.g6e.2xlarge for spot training job usage",
                "QuotaCode": "L-29512C0F",
                "Value": 1.0,
                "Adjustable": True,
            },
            {
                "QuotaName": "ml.g6e.2xlarge for training job usage",
                "QuotaCode": "L-D1AFBF6F",
                "Value": 1.0,
                "Adjustable": True,
            },
        ]
    )

    quota = verify_training_quota(
        service_quotas=client,
        instance_type="ml.g6e.2xlarge",
        instance_count=1,
        managed_spot=False,
    )

    assert quota["quota_name"] == "ml.g6e.2xlarge for training job usage"
    assert quota["quota_code"] == "L-D1AFBF6F"
    assert quota["managed_spot"] is False


def test_spot_only_quota_cannot_satisfy_on_demand_plan() -> None:
    """An on-demand submission must fail if only the Spot quota is found."""
    from lava.observability.smoke_guard import verify_training_quota

    client = _FakeServiceQuotas(
        [
            {
                "QuotaName": "ml.g6e.2xlarge for spot training job usage",
                "QuotaCode": "L-29512C0F",
                "Value": 1.0,
                "Adjustable": True,
            }
        ]
    )

    with pytest.raises(RuntimeError, match="Required SageMaker quota"):
        verify_training_quota(
            service_quotas=client,
            instance_type="ml.g6e.2xlarge",
            instance_count=1,
            managed_spot=False,
        )


def test_insufficient_exact_quota_fails_closed() -> None:
    """The correct quota must also have enough available instance capacity."""
    from lava.observability.smoke_guard import verify_training_quota

    client = _FakeServiceQuotas(
        [
            {
                "QuotaName": "ml.g6e.2xlarge for training job usage",
                "QuotaCode": "L-D1AFBF6F",
                "Value": 0.0,
                "Adjustable": True,
            }
        ]
    )

    with pytest.raises(RuntimeError, match="Insufficient SageMaker quota"):
        verify_training_quota(
            service_quotas=client,
            instance_type="ml.g6e.2xlarge",
            instance_count=1,
            managed_spot=False,
        )
