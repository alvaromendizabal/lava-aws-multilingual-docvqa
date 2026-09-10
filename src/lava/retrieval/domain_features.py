"""Label-blind document features and fixed-budget retrieval ablations.

These are research challengers, never an implicit change to frozen inference.
Native extraction is a fallible view of the PDF; tables and headings are signals,
not asserted semantic ground truth.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import fmean, median
from typing import Any

import pymupdf

from lava.evaluation.schemas import ReferenceRecord
from lava.readers.oracle_assets import document_aliases
from lava.retrieval.feature_research import evidence_metrics
from lava.retrieval.lexical import BM25Index, PageText, RetrievalQuery, tokenize
from lava.retrieval.research import BASELINE, METRICS, Rankings, select_candidate

COMPONENTS = ("body", "headings", "blocks", "tables", "numeric", "coverage", "adjacent")
POLICY: dict[str, Any] = {
    "page_budget": 5,
    "baseline_weight": 3.0,
    "feature_weight": 1.0,
    "rrf_k": 60,
    "body_vertical_interval": [0.10, 0.90],
    "heading_font_ratio": 1.25,
    "heading_top_fraction": 0.15,
    "table_strategies": ["lines_strict", "text"],
    "adjacent_seed_count": 2,
    "novel_page_baseline_keep": 4,
    "selection": "Existing conservative training-document gate; no automatic promotion",
}


@dataclass(frozen=True)
class DocumentPage:
    number: int
    native: str
    body: str
    headings: str
    blocks: tuple[str, ...]
    tables: str
    table_count: int
    table_strategy: str
    table_error: str | None = None


def extract_document(payload: bytes) -> tuple[DocumentPage, ...]:
    """Extract all physical pages; explicitly record failed table detection."""
    result = []
    with pymupdf.open(stream=payload, filetype="pdf") as pdf:
        if pdf.needs_pass or not len(pdf):
            raise ValueError("Research requires a readable, nonempty PDF")
        for number, page in enumerate(pdf, 1):
            native = page.get_text("text", sort=True).strip()
            blocks = [b for b in page.get_text("blocks", sort=True) if b[6] == 0]
            body = "\n".join(
                b[4] for b in blocks if 0.10 <= ((b[1] + b[3]) / 2) / page.rect.height <= 0.90
            )
            spans = [
                s
                for b in page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]
                for line in b.get("lines", [])
                for s in line["spans"]
                if s["text"].strip()
            ]
            typical = median(s["size"] for s in spans) if spans else 0
            headings = "\n".join(
                s["text"]
                for s in spans
                if s["size"] >= typical * 1.25 or s["bbox"][1] <= page.rect.height * 0.15
            )
            tables, strategy, error = [], "none", None
            try:
                for strategy in POLICY["table_strategies"]:
                    found = page.find_tables(strategy=strategy)
                    tables = [table.extract() for table in found.tables]
                    if tables:
                        break
            except (RuntimeError, ValueError, IndexError) as exc:
                error = type(exc).__name__
                tables = []
            table_text = "\n".join(
                " | ".join(str(cell or "") for cell in row) for table in tables for row in table
            )
            result.append(
                DocumentPage(
                    number,
                    native,
                    body,
                    headings,
                    tuple(b[4] for b in blocks),
                    table_text,
                    len(tables),
                    strategy,
                    error,
                )
            )
    return tuple(result)


def numeric_tokens(text: str) -> frozenset[str]:
    """Exact normalized digit strings; preserve separators instead of guessing locale."""
    return frozenset(re.findall(r"[+-]?\d+(?:[.,]\d+)*(?:%)?", unicodedata.normalize("NFKC", text)))


def _order(scores: dict[int, float], baseline: list[int]) -> list[int]:
    if set(scores) != set(baseline) or not all(math.isfinite(v) for v in scores.values()):
        raise ValueError("Feature scores must cover all pages and be finite")
    tie = {page: rank for rank, page in enumerate(baseline)}
    return sorted(scores, key=lambda page: (-scores[page], tie[page]))


def _field_scores(question: str, texts: Sequence[str]) -> dict[int, float]:
    pages = tuple(
        PageText(i, text, "ok" if text else "textless") for i, text in enumerate(texts, 1)
    )
    return dict(BM25Index(pages).rank(question))


def page_features(query: RetrievalQuery, pages: Sequence[DocumentPage]) -> dict[str, Any]:
    """Receive only the question and its complete PDF; no answers or gold pages."""
    if not pages or [p.number for p in pages] != list(range(1, len(pages) + 1)):
        raise ValueError("Features must retain every physical page exactly once in order")
    baseline_scores = _field_scores(query.question, [p.native for p in pages])
    baseline = sorted(baseline_scores, key=lambda p: (-baseline_scores[p], p))
    scores = {
        field: _field_scores(query.question, [getattr(p, field) for p in pages])
        for field in ("body", "headings", "tables")
    }
    all_blocks = [(p.number, block) for p in pages for block in p.blocks]
    block_scores = _field_scores(query.question, [b for _, b in all_blocks]) if all_blocks else {}
    scores["blocks"] = {p.number: 0.0 for p in pages}
    for index, (number, _) in enumerate(all_blocks, 1):
        scores["blocks"][number] = max(scores["blocks"][number], block_scores[index])
    digits = numeric_tokens(query.question)
    scores["numeric"] = {
        p.number: len(digits & numeric_tokens(p.native)) / len(digits) if digits else 0.0
        for p in pages
    }
    query_terms = set(tokenize(query.question))
    page_terms = {p.number: set(tokenize(p.native)) for p in pages}
    frequency: Counter[str] = Counter()
    for terms in page_terms.values():
        frequency.update(terms)
    weights = {
        term: math.log1p((len(pages) - frequency[term] + 0.5) / (frequency[term] + 0.5))
        for term in sorted(query_terms)
    }
    scores["coverage"] = {
        p.number: sum(weights[t] for t in sorted(query_terms & page_terms[p.number])) for p in pages
    }
    scores["adjacent"] = {
        p.number: sum(
            1 / (rank + 1) for rank, seed in enumerate(baseline[:2]) if abs(seed - p.number) == 1
        )
        if any(baseline_scores.values())
        else 0.0
        for p in pages
    }
    component_orders = {name: _order(scores[name], baseline) for name in COMPONENTS}
    active = {name: max(scores[name].values()) > min(scores[name].values()) for name in COMPONENTS}

    def fuse(names: Sequence[str]) -> list[int]:
        values = {p: 3.0 / (60 + i) for i, p in enumerate(baseline, 1)}
        for name in names:
            if active[name]:
                for rank, page in enumerate(component_orders[name], 1):
                    values[page] += 1.0 / (60 + rank)
        return _order(values, baseline)

    rankings = {BASELINE: baseline}
    rankings.update({"add_" + name: fuse([name]) for name in COMPONENTS})
    rankings["all_document_features"] = fuse(COMPONENTS)
    rankings.update(
        {"without_" + name: fuse([n for n in COMPONENTS if n != name]) for name in COMPONENTS}
    )
    chosen = baseline[:4]
    covered = set().union(*(page_terms[p] for p in chosen))
    uncovered = query_terms - covered
    novel_scores = {
        p: sum(weights[t] for t in sorted(uncovered & page_terms[p]))
        for p in baseline
        if p not in chosen
    }
    if novel_scores:
        novel = max(novel_scores, key=lambda p: (novel_scores[p], baseline_scores[p], -p))
        chosen = [*chosen, novel]
    rankings["query_complement_top4_plus1"] = chosen + [p for p in baseline if p not in chosen]
    return {"rankings": rankings, "active_features": active, "scores": scores}


def evaluate_ablation(rankings: Rankings, references: Sequence[ReferenceRecord]) -> dict[str, Any]:
    """Score fixed policies and make each selection without its held-out document."""
    if not references or len({r.question_id for r in references}) != len(references):
        raise ValueError("Evaluation needs unique, nonempty references")
    expected = {BASELINE, "all_document_features", "query_complement_top4_plus1"}
    expected |= {prefix + name for prefix in ("add_", "without_") for name in COMPONENTS}
    if set(rankings) != expected or any(len(v) != len(references) for v in rankings.values()):
        raise ValueError("Incomplete ablation catalog or question coverage")
    for i, ref in enumerate(references):
        pages = rankings[BASELINE][i]
        if sorted(pages) != list(range(1, len(pages) + 1)):
            raise ValueError("Baseline is not a complete physical page permutation")
        if not set(ref.evidence_pages) <= set(pages):
            raise ValueError("Reference pages exceed PDF bounds")
        if any(sorted(v[i]) != sorted(pages) for v in rankings.values()):
            raise ValueError("A candidate omitted or duplicated a physical page")
    aliases = document_aliases(tuple(references))

    def summarize(values: Sequence[dict[str, float]]) -> dict[str, float]:
        return {metric: fmean(v[metric] for v in values) for metric in METRICS}

    pooled, by_document = {}, {}
    for name, orders in rankings.items():
        values = [
            asdict(evidence_metrics(o, r.evidence_pages))
            for o, r in zip(orders, references, strict=True)
        ]
        pooled[name] = summarize(values)
        by_document[name] = {
            alias: summarize(
                [v for v, r in zip(values, references, strict=True) if r.document_id == doc]
            )
            for doc, alias in aliases.items()
        }
    folds, heldout = [], []
    names = list(rankings)
    for doc, alias in aliases.items():
        train = [i for i, r in enumerate(references) if r.document_id != doc]
        test = [i for i, r in enumerate(references) if r.document_id == doc]
        selected, audit = select_candidate(
            names,
            {n: [rankings[n][i] for i in train] for n in names},
            [references[i] for i in train],
            conservative=True,
        )
        values = [
            asdict(evidence_metrics(rankings[selected][i], references[i].evidence_pages))
            for i in test
        ]
        heldout.extend(values)
        folds.append(
            {
                "held_out_document": alias,
                "selected_candidate": selected,
                "selection_audit": audit,
                "metrics": summarize(values),
            }
        )
    return {
        "candidate_count": len(rankings),
        "question_count": len(references),
        "document_count": len(aliases),
        "pooled_diagnostics_not_selection": pooled,
        "by_document": by_document,
        "document_folds": folds,
        "document_isolated_selection": summarize(heldout),
        "baseline_selected_folds": sum(f["selected_candidate"] == BASELINE for f in folds),
        "globally_unique_ranking_signatures": len(
            {tuple(tuple(o) for o in orders) for orders in rankings.values()}
        ),
        "production_changed": False,
        "scope": "Post-hoc development audit on five previously examined PDFs; no independent test or new answer inference",
    }
