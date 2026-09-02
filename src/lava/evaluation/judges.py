"""Pluggable and cacheable semantic-equivalence judges."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from lava.evaluation.normalization import normalize_text, parse_decimal


class SemanticJudge(Protocol):
    """Interface required by the official-spec LAVA metric engine."""

    @property
    def identity(self) -> str:
        """Return a versioned identity for cache and provenance keys."""
        ...

    def equivalent(self, reference: str, prediction: str, *, language: str) -> bool:
        """Return whether two answer strings are semantically equivalent."""
        ...


@dataclass(frozen=True, slots=True)
class NormalizedExactJudge:
    """Deterministic judge for tests and non-semantic baseline diagnostics."""

    identity: str = "normalized-exact-v1"

    def equivalent(self, reference: str, prediction: str, *, language: str) -> bool:
        """Compare normalized text, then exact Decimal values when both are numeric."""
        del language
        if normalize_text(reference) == normalize_text(prediction):
            return True
        reference_number = parse_decimal(reference)
        prediction_number = parse_decimal(prediction)
        return (
            reference_number is not None
            and prediction_number is not None
            and reference_number == prediction_number
        )


@dataclass(slots=True)
class CachedSemanticJudge:
    """Memoize judge calls without changing the wrapped judge's semantics."""

    delegate: SemanticJudge
    _cache: dict[tuple[str, str, str, str], bool] = field(default_factory=dict)

    @property
    def identity(self) -> str:
        """Expose the wrapped judge identity."""
        return f"cached:{self.delegate.identity}"

    def equivalent(self, reference: str, prediction: str, *, language: str) -> bool:
        """Return a cached semantic-equivalence decision."""
        key = (self.delegate.identity, language, reference, prediction)
        if key not in self._cache:
            self._cache[key] = self.delegate.equivalent(
                reference,
                prediction,
                language=language,
            )
        return self._cache[key]

    @property
    def cache_size(self) -> int:
        """Return the number of unique judged pairs."""
        return len(self._cache)
