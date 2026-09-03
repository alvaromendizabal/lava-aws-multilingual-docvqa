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
