"""Leakage-safe lexical feature research for the LAVA retrieval diagnostic.

The production retriever remains intentionally small.  This module owns a much
broader research-only candidate space so feature discovery can be exhaustive
without silently changing the inference contract.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BM25FeatureSpec:
    """One scalar query-page scoring feature in the prespecified BM25 search grid."""

    representation: str
    k1: float
    b: float

    @property
    def name(self) -> str:
        return f"bm25|{self.representation}|k1={self.k1:g}|b={self.b:g}"


@dataclass(frozen=True, slots=True)
class FusionSpec:
    """One conservative rank-fusion policy built only from fixed lexical views."""

    method: str
    systems: tuple[str, ...]
    weights: tuple[float, ...]
    rrf_k: int | None = None

    def __post_init__(self) -> None:
        if self.method not in {"rrf", "borda"}:
            raise ValueError("Unknown fusion method")
        if len(self.systems) < 2 or len(self.systems) != len(self.weights):
            raise ValueError("Fusion systems and weights must align")
        if any(weight <= 0 or not math.isfinite(weight) for weight in self.weights):
            raise ValueError("Fusion weights must be finite and positive")
        if self.method == "rrf" and (self.rrf_k is None or self.rrf_k < 1):
            raise ValueError("RRF requires a positive rank constant")
        if self.method == "borda" and self.rrf_k is not None:
            raise ValueError("Borda must not carry an RRF rank constant")

    @property
    def name(self) -> str:
        weights = ",".join(f"{weight:g}" for weight in self.weights)
        suffix = f"|k={self.rrf_k}" if self.rrf_k is not None else ""
        return f"{self.method}|{'+'.join(self.systems)}|w={weights}{suffix}"


@dataclass(frozen=True, slots=True)
class ExplorationSpec:
    """Keep baseline pages and spend the remaining budget on challenger consensus."""

    challengers: tuple[str, ...]
    baseline_keep: int

    def __post_init__(self) -> None:
        if not self.challengers or not 1 <= self.baseline_keep <= 4:
            raise ValueError("Exploration policy needs challengers and baseline_keep in [1, 4]")

    @property
    def name(self) -> str:
        novel = 5 - self.baseline_keep
        return f"explore|B{self.baseline_keep}+{'+'.join(self.challengers)}{novel}"


@dataclass(frozen=True, slots=True)
class EvidenceMetrics:
    """Upper-bound evidence coverage metrics before downstream answer generation."""

    oracle_grounding_f1_at_k: float
    all_evidence_at_k: float
    recall_at_k: float
    ndcg_at_k: float


REPRESENTATIONS: Mapping[str, tuple[str, ...]] = {
    "word": ("word",),
    "char2": ("c2",),
    "char3": ("c3",),
    "char4": ("c4",),
    "char5": ("c5",),
    "word_char23": ("word", "c2", "c3"),
    "word_char234": ("word", "c2", "c3", "c4"),
    "char23": ("c2", "c3"),
    "char234": ("c2", "c3", "c4"),
    "flat3": ("f3",),
    "flat4": ("f4",),
}
K1_GRID = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0)
B_GRID = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
FIXED_FUSION_SYSTEMS = ("B", "C", "F", "W")


def bm25_feature_grid() -> tuple[BM25FeatureSpec, ...]:
    """Return the 1,089 prespecified lexical query-page score features."""
    return tuple(
        BM25FeatureSpec(representation, k1, b)
        for representation in REPRESENTATIONS
        for k1 in K1_GRID
        for b in B_GRID
    )


def fusion_grid() -> tuple[FusionSpec, ...]:
    """Return 472 prespecified weighted RRF/Borda policies over fixed views.

    Exploration policies are evaluated separately because they are selectors rather
    than score fusions.  Keeping them separate makes candidate accounting explicit.
    """
    specs: list[FusionSpec] = []
    groups = (
        ("B", "C"),
        ("B", "F"),
        ("B", "W"),
        ("B", "C", "F"),
        ("B", "C", "W"),
        ("B", "F", "W"),
        ("B", "C", "F", "W"),
    )
    for systems in groups:
        if len(systems) == 2:
            weights = tuple((base, 1.0) for base in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0))
        elif len(systems) == 3:
            weights = tuple((base, 1.0, 1.0) for base in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)) + (
                (2.0, 2.0, 1.0),
                (2.0, 1.0, 2.0),
                (3.0, 2.0, 1.0),
                (3.0, 1.0, 2.0),
            )
        else:
            weights = tuple((base, 1.0, 1.0, 1.0) for base in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)) + (
                (2.0, 2.0, 1.0, 1.0),
                (2.0, 1.0, 2.0, 1.0),
                (2.0, 1.0, 1.0, 2.0),
                (3.0, 2.0, 1.0, 1.0),
                (3.0, 1.0, 2.0, 1.0),
            )
        for weight in weights:
            specs.extend(FusionSpec("rrf", systems, weight, k) for k in (1, 5, 10, 20, 30, 60, 100))
            specs.append(FusionSpec("borda", systems, weight))
    return tuple(specs)


def exploration_grid() -> tuple[ExplorationSpec, ...]:
    """Return the 21 baseline-preserving exploration selectors used in research."""
    challenger_groups = (
        ("C",),
        ("F",),
        ("W",),
        ("C", "F"),
        ("C", "W"),
        ("F", "W"),
        ("C", "F", "W"),
    )
    return tuple(
        ExplorationSpec(challengers, baseline_keep)
        for challengers in challenger_groups
        for baseline_keep in (2, 3, 4)
    )


def exploration_order(
    baseline: Sequence[int],
    challengers: Sequence[Sequence[int]],
    *,
    baseline_keep: int,
    page_budget: int = 5,
    consensus_k: int = 20,
) -> tuple[int, ...]:
    """Preserve top baseline pages, then add challenger-consensus exploration pages."""
    if not 1 <= baseline_keep < page_budget or not challengers or consensus_k < 1:
        raise ValueError("Invalid exploration policy")
    page_set = set(baseline)
    if any(set(order) != page_set or len(order) != len(page_set) for order in challengers):
        raise ValueError("Every challenger must rank the same pages as the baseline")
    chosen = list(baseline[:baseline_keep])
    scores = {page: 0.0 for page in page_set}
    for order in challengers:
        for position, page in enumerate(order, 1):
            scores[page] += 1.0 / (consensus_k + position)
    for page in sorted(page_set, key=lambda value: (-scores[value], value)):
        if page not in chosen:
            chosen.append(page)
        if len(chosen) >= page_budget:
            break
    tail = [page for page in baseline if page not in chosen]
    return tuple(chosen + tail)


def _primitive_tokens(text: str) -> dict[str, Counter[str]]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    segments = re.findall(r"[^\W_]+", normalized)
    flat = "".join(segments)
    values: dict[str, Counter[str]] = {"word": Counter("w:" + segment for segment in segments)}
    for size in (2, 3, 4, 5):
        values[f"c{size}"] = Counter(
            f"c{size}:" + segment[start : start + size]
            for segment in segments
            for start in range(max(0, len(segment) - size + 1))
        )
    for size in (3, 4):
        values[f"f{size}"] = Counter(
            f"f{size}:" + flat[start : start + size]
            for start in range(max(0, len(flat) - size + 1))
        )
    return values


def feature_tokens(text: str, representation: str) -> Counter[str]:
    """Materialize one documented lexical representation without label access."""
    try:
        parts = REPRESENTATIONS[representation]
    except KeyError as error:
        raise ValueError(f"Unknown representation: {representation}") from error
    primitives = _primitive_tokens(text)
    result: Counter[str] = Counter()
    for part in parts:
        result.update(primitives[part])
    return result


def bm25_scores(
    question: str,
    pages: Sequence[tuple[int, str]],
    spec: BM25FeatureSpec,
) -> tuple[tuple[int, float], ...]:
    """Score every physical page for one feature specification."""
    if not pages or tuple(number for number, _ in pages) != tuple(range(1, len(pages) + 1)):
        raise ValueError("Pages must cover the complete physical PDF in order")
    query = feature_tokens(question, spec.representation)
    counts = [feature_tokens(text, spec.representation) for _, text in pages]
    lengths = [sum(count.values()) for count in counts]
    average = sum(lengths) / len(lengths)
    document_frequency: Counter[str] = Counter()
    for count in counts:
        document_frequency.update(count.keys())
    scored: list[tuple[int, float]] = []
    for (number, _), count, length in zip(pages, counts, lengths, strict=True):
        score = 0.0
        if average:
            normalization = spec.k1 * (1 - spec.b + spec.b * length / average)
            for term in sorted(query):
                frequency = count[term]
                if frequency:
                    df = document_frequency[term]
                    idf = math.log1p((len(pages) - df + 0.5) / (df + 0.5))
                    score += idf * frequency * (spec.k1 + 1) / (frequency + normalization)
        scored.append((number, score))
    return tuple(sorted(scored, key=lambda item: (-item[1], item[0])))


def ranking_signature(rankings: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """Create the label-blind signature used to reject redundant candidates."""
    return tuple(tuple(ranking) for ranking in rankings)


def deduplicate_rankings(
    candidates: Iterable[tuple[str, Sequence[Sequence[int]]]],
) -> tuple[dict[str, tuple[tuple[int, ...], ...]], int]:
    """Reject candidates with identical rankings before looking at gold evidence."""
    retained: dict[str, tuple[tuple[int, ...], ...]] = {}
    seen: set[tuple[tuple[int, ...], ...]] = set()
    rejected = 0
    for name, rankings in candidates:
        signature = ranking_signature(rankings)
        if signature in seen:
            rejected += 1
            continue
        seen.add(signature)
        retained[name] = signature
    return retained, rejected


def fuse_ranks(
    rankings: Sequence[Sequence[int]],
    weights: Sequence[float],
    *,
    method: str,
    rrf_k: int | None = None,
) -> tuple[int, ...]:
    """Fuse complete page rankings with stable page-number tie breaking."""
    if not rankings or len(rankings) != len(weights):
        raise ValueError("Rankings and weights must be nonempty and aligned")
    page_set = set(rankings[0])
    if any(set(ranking) != page_set or len(ranking) != len(page_set) for ranking in rankings):
        raise ValueError("Every system must rank every physical page exactly once")
    if any(weight <= 0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("Weights must be finite and positive")
    scores = {page: 0.0 for page in page_set}
    if method == "rrf":
        if rrf_k is None or rrf_k < 1:
            raise ValueError("RRF requires a positive rank constant")
        for ranking, weight in zip(rankings, weights, strict=True):
            for position, page in enumerate(ranking, 1):
                scores[page] += weight / (rrf_k + position)
    elif method == "borda":
        if rrf_k is not None:
            raise ValueError("Borda must not receive an RRF rank constant")
        for ranking, weight in zip(rankings, weights, strict=True):
            count = len(ranking)
            for position, page in enumerate(ranking):
                scores[page] += weight * (count - position) / count
    else:
        raise ValueError("Unknown fusion method")
    return tuple(sorted(page_set, key=lambda page: (-scores[page], page)))


def evidence_metrics(order: Sequence[int], gold: Iterable[int], *, k: int = 5) -> EvidenceMetrics:
    """Compute leakage-free retrieval diagnostics after a ranking has been frozen."""
    ranking = tuple(order)
    gold_set = frozenset(gold)
    if not gold_set or k < 1 or len(set(ranking)) != len(ranking) or not gold_set.issubset(ranking):
        raise ValueError("Ranking/gold coverage is invalid")
    top = ranking[:k]
    hits = len(gold_set.intersection(top))
    recall = hits / len(gold_set)
    oracle_f1 = 2 * hits / (hits + len(gold_set)) if hits else 0.0
    dcg = sum(
        1.0 / math.log2(position + 2) for position, page in enumerate(top) if page in gold_set
    )
    ideal = sum(1.0 / math.log2(position + 2) for position in range(min(k, len(gold_set))))
    return EvidenceMetrics(
        oracle_grounding_f1_at_k=oracle_f1,
        all_evidence_at_k=float(gold_set.issubset(top)),
        recall_at_k=recall,
        ndcg_at_k=dcg / ideal,
    )
