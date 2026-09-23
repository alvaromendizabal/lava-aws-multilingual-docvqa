"""Conservative normalization for model-emitted physical page citations.

Only serialization-equivalent drift is normalized. The helper accepts JSON integers
and exact positive decimal strings such as ``"12"``. It deliberately rejects floats,
booleans, signed strings, ranges, prose, zero, duplicate pages, and unavailable pages.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

_DECIMAL_PAGE = re.compile(r"[1-9]\d*")


@dataclass(frozen=True, slots=True)
class EvidencePageNormalization:
    """Normalized physical pages plus an auditable count of structural repairs."""

    pages: tuple[int, ...]
    converted_decimal_strings: int

    @property
    def changed(self) -> bool:
        """Whether the input required a lossless string-to-integer conversion."""
        return self.converted_decimal_strings > 0


def normalize_model_evidence_pages(
    value: Any,
    *,
    allowed_pages: Iterable[int] | None = None,
) -> EvidencePageNormalization:
    """Normalize a model-emitted JSON page array without changing its meaning.

    Args:
        value: The decoded JSON value for ``evidence_pages``.
        allowed_pages: Optional physical-page allowlist for an already selected context.

    Raises:
        TypeError: The root is not a JSON array.
        ValueError: A page is ambiguous, non-positive, duplicated, or unavailable.
    """
    if not isinstance(value, list):
        raise TypeError("Evidence pages must be a JSON array")

    pages: list[int] = []
    converted = 0
    for candidate in value:
        if type(candidate) is int:
            page = candidate
        elif isinstance(candidate, str) and _DECIMAL_PAGE.fullmatch(candidate.strip()):
            page = int(candidate.strip())
            converted += 1
        else:
            raise ValueError(
                "Evidence pages must contain positive JSON integers or exact decimal strings"
            )
        if page < 1:
            raise ValueError(f"Evidence page must be positive, received {page}")
        pages.append(page)

    if len(set(pages)) != len(pages):
        raise ValueError("Evidence pages must be unique")

    if allowed_pages is not None:
        allowed = set(allowed_pages)
        unavailable = sorted(set(pages) - allowed)
        if unavailable:
            raise ValueError(f"Evidence pages are outside the supplied context: {unavailable}")

    return EvidencePageNormalization(
        pages=tuple(sorted(pages)),
        converted_decimal_strings=converted,
    )
