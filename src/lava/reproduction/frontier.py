"""Small, testable primitives used by the LAVA frontier reproduction benchmark.

This module intentionally contains no private questions, answers, predictions, or credentials.
It captures the public experimental contract: fixed reader arms, retrieval fusion helpers,
lossless evidence-page normalization, and parser-independent generation-cache identity.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ReproductionArm:
    key: str
    reader: str
    retrieval: str
    reasoning: str


ARMS: tuple[ReproductionArm, ...] = (
    ReproductionArm("A", "Qwen3.5-9B BF16", "BM25 top-5", "direct"),
    ReproductionArm("B", "Qwen3.5-9B BF16", "BM25 top-10", "direct"),
    ReproductionArm("C", "Qwen3.6-27B NF4", "UIT-style BM25Plus + E5 adaptive", "direct"),
    ReproductionArm("D", "Qwen3.6-27B NF4", "BM25 + ColQwen adaptive", "direct"),
    ReproductionArm("E", "Qwen3.6-27B NF4", "BM25 + ColQwen adaptive", "evidence decomposition"),
    ReproductionArm("F", "Qwen3.6-27B NF4", "BM25 + ColQwen adaptive", "cross-modal conflict reread"),
)


def normalize_evidence_pages(value: Any, allowed: Sequence[int]) -> list[int]:
    """Normalize only lossless physical-page identifiers.

    Digit-only strings such as ``"11"`` are equivalent to integer page 11 and are accepted.
    Floats, decimal strings, booleans, zero, duplicates, and unavailable pages are rejected.
    """
    if not isinstance(value, list):
        raise ValueError("evidence_pages must be a list")
    normalized: list[int] = []
    for item in value:
        if type(item) is int:
            page = item
        elif isinstance(item, str) and re.fullmatch(r"\s*[1-9]\d*\s*", item):
            page = int(item.strip())
        else:
            raise ValueError("evidence_pages must contain integer page identifiers")
        normalized.append(page)
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate evidence page")
    if not set(normalized).issubset(set(allowed)):
        raise ValueError("unavailable evidence page")
    return sorted(normalized)


def generation_semantics(request: Mapping[str, Any]) -> dict[str, Any]:
    """Strip parser/report implementation identity from a deterministic generation request."""
    ignored = {"implementation", "generation_contract_version", "parser_contract_version", "model_runtime_sha256"}
    return {key: value for key, value in request.items() if key not in ignored}


def reciprocal_rank_fusion(*rankings: Sequence[int], k: int = 60) -> list[tuple[int, float]]:
    """Fuse unique page rankings using standard reciprocal-rank fusion."""
    if k <= 0 or not rankings:
        raise ValueError("invalid RRF configuration")
    scores: dict[int, float] = {}
    for ranking in rankings:
        if len(ranking) != len(set(ranking)):
            raise ValueError("ranking contains duplicate pages")
        for rank, page in enumerate(ranking, start=1):
            scores[page] = scores.get(page, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def adaptive_page_cap(total_pages: int) -> int:
    """Reproduce the public 2026 reference cap schedule used in our documented adaptation."""
    if total_pages < 1:
        raise ValueError("document must contain at least one page")
    return 10 if total_pages > 20 else 8 if total_pages > 10 else 7


def set_f1(reference: Iterable[int], prediction: Iterable[int]) -> float:
    reference_set, prediction_set = set(reference), set(prediction)
    if not reference_set and not prediction_set:
        return 1.0
    if not reference_set or not prediction_set:
        return 0.0
    return 2.0 * len(reference_set & prediction_set) / (len(reference_set) + len(prediction_set))
