"""Fail-closed validation for the first one-question GPU smoke job."""

from __future__ import annotations

import time
from collections.abc import Mapping

from botocore.exceptions import ClientError  # type: ignore[import-untyped]


def _require_int(plan: Mapping[str, object], key: str) -> int:
    value = plan.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        message = f"Plan field {key!r} must be an integer."
        raise TypeError(message)
    return value


def validate_first_smoke_plan(
    plan: Mapping[str, object],
) -> None:
    """Fail closed unless a paid one-question GPU smoke plan is safely bounded."""
    errors: list[str] = []

    model_key = plan.get("model_key")
    if not isinstance(model_key, str) or not model_key.strip():
        errors.append("model_key must be a non-empty string")

    instance_type = plan.get("instance_type")

    if not isinstance(instance_type, str) or not instance_type.strip():
        errors.append("instance_type must be a non-empty string")
    elif (
        instance_type,
        False,
    ) not in _EXACT_TRAINING_QUOTA_CODES:
        errors.append(
            f"instance_type has no pinned on-demand SageMaker quota binding: {instance_type!r}"
        )

    if (
        _require_int(
            plan,
            "limit",
        )
        != 1
    ):
        errors.append("limit must equal 1 for a paid smoke")

    if (
        _require_int(
            plan,
            "instance_count",
        )
        != 1
    ):
        errors.append("instance_count must equal 1")

    if (
        _require_int(
            plan,
            "max_runtime_seconds",
        )
        > 3600
    ):
        errors.append("max_runtime_seconds must be <= 3600")

    if (
        _require_int(
            plan,
            "max_wait_seconds",
        )
        > 7200
    ):
        errors.append("max_wait_seconds must be <= 7200")
    if _require_int(plan, "max_pending_seconds") != 86400:
        errors.append("max_pending_seconds must equal 86400 for the first smoke")

    if plan.get("creates_endpoint") is not False:
        errors.append("creates_endpoint must be false")

    if plan.get("managed_spot") is not False:
        errors.append("managed_spot must be false for the certified frontier smoke path")

    if errors:
        raise RuntimeError("Unsafe smoke plan: " + "; ".join(errors))


_EXACT_TRAINING_QUOTA_CODES: dict[tuple[str, bool], str] = {
    ("ml.g5.2xlarge", False): "L-2D6DEB3C",
    ("ml.g5.2xlarge", True): "L-CAEE7DB7",
    ("ml.g6e.2xlarge", False): "L-D1AFBF6F",
    ("ml.g6e.2xlarge", True): "L-29512C0F",
    ("ml.g7e.12xlarge", False): "L-99850E94",
    ("ml.g7e.48xlarge", False): "L-BE072D49",
    ("ml.p5en.48xlarge", False): "L-1E48384D",
    ("ml.p6-b200.48xlarge", False): "L-60EA3D74",
    ("ml.p6-b300.48xlarge", False): "L-82BE9A32",
}

_RETRIABLE_SERVICE_QUOTA_ERRORS = frozenset(
    {
        "TooManyRequestsException",
        "ThrottlingException",
        "Throttling",
        "RequestLimitExceeded",
    }
)

_MAX_SERVICE_QUOTA_ATTEMPTS = 5


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

    quota_code = _EXACT_TRAINING_QUOTA_CODES.get((instance_type, managed_spot))
    if quota_code is None:
        message = (
            "No immutable SageMaker quota-code binding exists for "
            f"instance_type={instance_type!r}, managed_spot={managed_spot!r}. "
            "Refusing to submit a paid training job."
        )
        raise RuntimeError(message)

    get_service_quota = getattr(service_quotas, "get_service_quota", None)
    if not callable(get_service_quota):
        message = "Service Quotas client does not expose get_service_quota()."
        raise TypeError(message)

    response: object | None = None
    attempts_used = 0

    for attempt in range(1, _MAX_SERVICE_QUOTA_ATTEMPTS + 1):
        attempts_used = attempt
        try:
            response = get_service_quota(
                ServiceCode="sagemaker",
                QuotaCode=quota_code,
            )
            break
        except ClientError as exc:
            error_code = str(exc.response.get("Error", {}).get("Code", ""))
            retryable = error_code in _RETRIABLE_SERVICE_QUOTA_ERRORS

            if not retryable or attempt >= _MAX_SERVICE_QUOTA_ATTEMPTS:
                raise

            time.sleep(float(2 ** (attempt - 1)))

    if not isinstance(response, dict):
        message = f"SageMaker quota {expected_name!r} returned an invalid response."
        raise TypeError(message)

    quota = response.get("Quota")
    if quota is None:
        message = (
            f"Required SageMaker quota {expected_name!r} was not returned. "
            "Refusing to submit a paid training job."
        )
        raise RuntimeError(message)
    if not isinstance(quota, dict):
        message = f"SageMaker quota {expected_name!r} returned malformed quota metadata."
        raise TypeError(message)

    actual_name = quota.get("QuotaName")
    if actual_name != expected_name:
        message = (
            "SageMaker quota code/name mismatch: "
            f"quota_code={quota_code!r}, "
            f"expected_name={expected_name!r}, "
            f"actual_name={actual_name!r}. "
            "Refusing to submit a paid training job."
        )
        raise RuntimeError(message)

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
        "quota_code": quota_code,
        "value": numeric_value,
        "adjustable": quota.get("Adjustable"),
        "managed_spot": managed_spot,
        "required_instance_count": instance_count,
        "api_attempts": attempts_used,
    }
