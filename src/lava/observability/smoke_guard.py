"""Fail-closed validation for the first one-question GPU smoke job."""

from __future__ import annotations

from collections.abc import Mapping

_ALLOWED_MODEL_KEY = "qwen35_4b_fused_direct"
_ALLOWED_INSTANCE_TYPE = "ml.g6e.2xlarge"


def _require_int(plan: Mapping[str, object], key: str) -> int:
    value = plan.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        message = f"Plan field {key!r} must be an integer."
        raise TypeError(message)
    return value


def validate_first_smoke_plan(plan: Mapping[str, object]) -> None:
    """Validate the bounded first experiment before any paid submission."""
    errors: list[str] = []
    if plan.get("model_key") != _ALLOWED_MODEL_KEY:
        errors.append(f"model_key must be {_ALLOWED_MODEL_KEY!r}")
    if plan.get("instance_type") != _ALLOWED_INSTANCE_TYPE:
        errors.append(f"instance_type must be {_ALLOWED_INSTANCE_TYPE!r}")
    if _require_int(plan, "limit") != 1:
        errors.append("limit must equal 1")
    if _require_int(plan, "instance_count") != 1:
        errors.append("instance_count must equal 1")
    if _require_int(plan, "max_runtime_seconds") > 3600:
        errors.append("max_runtime_seconds must be at most 3600")
    if bool(plan.get("creates_endpoint")):
        errors.append("creates_endpoint must be false")
    if bool(plan.get("managed_spot")):
        errors.append("managed_spot must be false for the first infrastructure smoke test")
    if errors:
        message = "Unsafe first-smoke plan: " + "; ".join(errors)
        raise RuntimeError(message)


def required_training_quota_name(
    instance_type: str,
    *,
    managed_spot: bool,
) -> str:
    """Return the exact SageMaker Training quota name required by a plan."""
    suffix = "spot training job usage" if managed_spot else "training job usage"
    return f"{instance_type} for {suffix}"


def verify_training_quota(
    *,
    service_quotas: object,
    instance_type: str,
    instance_count: int,
    managed_spot: bool,
) -> dict[str, object]:
    """Fail closed unless the exact required SageMaker Training quota is sufficient."""
    if not instance_type:
        message = "instance_type must be non-empty."
        raise ValueError(message)
    if instance_count < 1:
        message = "instance_count must be at least one."
        raise ValueError(message)

    expected_name = required_training_quota_name(
        instance_type,
        managed_spot=managed_spot,
    )

    get_paginator = getattr(service_quotas, "get_paginator", None)
    if not callable(get_paginator):
        message = "Service Quotas client does not expose get_paginator()."
        raise TypeError(message)

    paginator = get_paginator("list_service_quotas")
    for page in paginator.paginate(ServiceCode="sagemaker"):
        quotas = page.get("Quotas", [])
        if not isinstance(quotas, list):
            continue

        for quota in quotas:
            if not isinstance(quota, dict):
                continue
            if quota.get("QuotaName") != expected_name:
                continue

            value = quota.get("Value")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                message = f"SageMaker quota {expected_name!r} has no numeric value."
                raise TypeError(message)

            numeric_value = float(value)
            if numeric_value < float(instance_count):
                message = (
                    f"Insufficient SageMaker quota {expected_name!r}: "
                    f"required={instance_count}, available={numeric_value}."
                )
                raise RuntimeError(message)

            return {
                "found": True,
                "quota_name": expected_name,
                "quota_code": quota.get("QuotaCode"),
                "value": numeric_value,
                "adjustable": quota.get("Adjustable"),
                "managed_spot": managed_spot,
                "required_instance_count": instance_count,
            }

    message = (
        f"Required SageMaker quota {expected_name!r} was not found. "
        "Refusing to submit a paid training job."
    )
    raise RuntimeError(message)
