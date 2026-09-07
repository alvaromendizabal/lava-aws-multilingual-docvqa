"""Deterministic, document-scoped page retrieval without answer or evidence inputs."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PageText:
    """One physical PDF page, including pages with no extractable native text."""

    number: int
    text: str
    status: str = "ok"


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """The complete retrieval input: identity, source document, and question only."""

    question_id: str
    document_id: str
    question: str


def tokenize(text: str) -> tuple[str, ...]:
    """NFKC/casefold words plus within-word character bigrams and trigrams.

    Character features support Japanese without a whitespace tokenizer or learned
    vocabulary. Vietnamese diacritics remain intact. No label-derived stopwords,
    query expansion, stemming, or language-specific tuning is applied.
    """
    segments = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())
    tokens: list[str] = []
    for segment in segments:
        tokens.append("w:" + segment)
        for size in (2, 3):
            tokens.extend(
                f"c{size}:" + segment[start : start + size]
                for start in range(len(segment) - size + 1)
            )
    return tuple(tokens)


class BM25Index:
    """Exact sparse BM25 over every page in one PDF, with stable page-number ties."""

    def __init__(self, pages: tuple[PageText, ...], *, k1: float = 1.2, b: float = 0.75):
        if not pages or tuple(page.number for page in pages) != tuple(range(1, len(pages) + 1)):
            raise ValueError("Pages must cover the complete PDF in physical one-indexed order")
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 requires finite k1 > 0 and b in [0, 1]")
        if any(page.status not in {"ok", "textless", "extraction_error"} for page in pages):
            raise ValueError("Unknown extraction status")
        if any(page.text and page.status != "ok" for page in pages):
            raise ValueError("Unsuccessful pages must not carry unverified text")
        self.pages, self.k1, self.b = pages, k1, b
        self.counts = tuple(Counter(tokenize(page.text)) for page in pages)
        self.lengths = tuple(sum(count.values()) for count in self.counts)
        self.average_length = sum(self.lengths) / len(pages)
        self.document_frequency: Counter[str] = Counter()
        for count in self.counts:
            self.document_frequency.update(count.keys())

    def rank(self, question: str) -> tuple[tuple[int, float], ...]:
        """Score a question with no access to gold answers, citations, or other PDFs."""
        # Sorting makes floating-point summation stable across PYTHONHASHSEED values.
        query_terms = sorted(set(tokenize(question)))
        scores: list[tuple[int, float]] = []
        for page, count, length in zip(self.pages, self.counts, self.lengths, strict=True):
            score = 0.0
            if self.average_length:
                normalization = self.k1 * (1 - self.b + self.b * length / self.average_length)
                for term in query_terms:
                    frequency = count[term]
                    if frequency:
                        df = self.document_frequency[term]
                        idf = math.log1p((len(self.pages) - df + 0.5) / (df + 0.5))
                        score += idf * frequency * (self.k1 + 1) / (frequency + normalization)
            scores.append((page.number, score))
        return tuple(sorted(scores, key=lambda item: (-item[1], item[0])))
