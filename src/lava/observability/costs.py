"""Conservative, operator-supplied cloud cost guards."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Maximum charge estimate derived from an operator-provided hourly ceiling."""

    hourly_usd_ceiling: float
    max_runtime_seconds: int
    instance_count: int
    contingency_factor: float
    estimated_maximum_usd: float

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def estimate_maximum_cost(
    *,
    hourly_usd_ceiling: float,
    max_runtime_seconds: int,
    instance_count: int = 1,
    contingency_factor: float = 1.25,
) -> CostEstimate:
    """Estimate a conservative upper bound without claiming a live AWS price quote."""
    numeric_values = (hourly_usd_ceiling, contingency_factor)
    if any(not math.isfinite(value) or value <= 0 for value in numeric_values):
        message = "Cost rates and contingency factor must be finite and greater than zero."
        raise ValueError(message)
    if max_runtime_seconds <= 0:
        message = "max_runtime_seconds must be greater than zero."
        raise ValueError(message)
    if instance_count <= 0:
        message = "instance_count must be greater than zero."
        raise ValueError(message)
    hours = max_runtime_seconds / 3600.0
    maximum = hourly_usd_ceiling * hours * instance_count * contingency_factor
    return CostEstimate(
        hourly_usd_ceiling=hourly_usd_ceiling,
        max_runtime_seconds=max_runtime_seconds,
        instance_count=instance_count,
        contingency_factor=contingency_factor,
        estimated_maximum_usd=round(maximum, 4),
    )


def enforce_cost_cap(estimate: CostEstimate, *, maximum_allowed_usd: float) -> None:
    """Fail closed when a conservative estimate exceeds the operator's cap."""
    if not math.isfinite(maximum_allowed_usd) or maximum_allowed_usd <= 0:
        message = "maximum_allowed_usd must be finite and greater than zero."
        raise ValueError(message)
    if estimate.estimated_maximum_usd > maximum_allowed_usd:
        message = (
            "Conservative cost estimate exceeds the allowed cap: "
            f"estimated=${estimate.estimated_maximum_usd:.2f}, "
            f"cap=${maximum_allowed_usd:.2f}."
        )
        raise RuntimeError(message)
