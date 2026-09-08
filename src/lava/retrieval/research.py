"""Executable, checkpointed feature research with document-isolated selection.

This audit fixes every fusion view at k1=1.2, b=0.75 and records the complete
selection rule in source. It supersedes the early exploratory aggregate; it
cannot change the production retriever or claim an independent test result.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any

from lava.evaluation.schemas import ReferenceRecord
from lava.evaluation.semantic import digest, encode
from lava.readers.oracle_assets import document_aliases
from lava.readers.runtime_logging import RuntimeEventLogger
from lava.retrieval.feature_research import (
    REPRESENTATIONS,
    BM25FeatureSpec,
    bm25_feature_grid,
    evidence_metrics,
    exploration_grid,
    exploration_order,
    feature_tokens,
    fuse_ranks,
    fusion_grid,
)
from lava.retrieval.lexical import PageText, RetrievalQuery
from lava.retrieval.pipeline import ObjectStore, stage

BASELINE = BM25FeatureSpec("word_char23", 1.2, 0.75).name
FUSION_VIEWS = {"B": "word_char23", "C": "char234", "F": "flat3", "W": "word"}
METRICS = ("oracle_grounding_f1_at_k", "all_evidence_at_k", "recall_at_k", "ndcg_at_k")
Rankings = dict[str, list[list[int]]]


class PreparedBM25:
    """Reuse label-free token statistics across the 99 parameter pairs per view."""

    def __init__(self, pages: Sequence[PageText], representation: str):
        if not pages or [p.number for p in pages] != list(range(1, len(pages) + 1)):
            raise ValueError("Research must retain every physical page in order")
        self.representation = representation
        self.counts = [feature_tokens(page.text, representation) for page in pages]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.average = fmean(self.lengths)
        df: Counter[str] = Counter()
        for count in self.counts:
            df.update(count.keys())
        self.idf = {
            term: math.log1p((len(pages) - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def rank(self, question: str, k1: float, b: float) -> list[int]:
        terms = sorted(feature_tokens(question, self.representation))
        scored = []
        for page, (counts, length) in enumerate(zip(self.counts, self.lengths, strict=True), 1):
            score = 0.0
            if self.average:
                normalization = k1 * (1 - b + b * length / self.average)
                for term in terms:
                    frequency = counts[term]
                    if frequency:
                        score += self.idf[term] * frequency * (k1 + 1) / (frequency + normalization)
            if not math.isfinite(score):
                raise ValueError("Non-finite query-page feature")
            scored.append((page, score))
        return [page for page, _ in sorted(scored, key=lambda item: (-item[1], item[0]))]


def research_contract(root: Path, retrieval_contract: str) -> dict[str, Any]:
    paths = (
        "src/lava/retrieval/feature_research.py",
        "src/lava/retrieval/research.py",
        "scripts/research_retrieval.py",
    )
    return {
        "schema_version": 1,
        "source_retrieval_contract": retrieval_contract,
        "source_sha256": {path: digest((root / path).read_bytes()) for path in paths},
        "fusion_views": FUSION_VIEWS,
        "fusion_k1": 1.2,
        "fusion_b": 0.75,
        "page_budget": 5,
        "selection": "Document-macro grounding F1, complete evidence, recall, nDCG; then catalog order",
        "conservative_gate": {
            "minimum_training_document_mean_f1_gain": 0.02,
            "minimum_improved_training_documents": 2,
            "maximum_regressed_training_documents": 0,
        },
        "scope": "Post-hoc development audit; previously examined five PDFs, no independent test",
    }


def build_rankings(
    queries: Sequence[RetrievalQuery],
    documents: Mapping[str, Sequence[PageText]],
    store: ObjectStore,
    contract: dict[str, Any],
    logger: RuntimeEventLogger,
) -> tuple[Rankings, dict[str, int]]:
    """Persist each full feature family before labels enter any scoring function."""
    if not queries or len({q.question_id for q in queries}) != len(queries):
        raise ValueError("Research requires unique, nonempty queries")
    if {q.document_id for q in queries} != set(documents):
        raise ValueError("Research query/PDF coverage differs")
    identity = {
        "contract": contract,
        "queries_sha256": digest(encode([asdict(query) for query in queries])),
        "documents_sha256": digest(
            encode({doc: [asdict(page) for page in pages] for doc, pages in documents.items()})
        ),
    }
    rankings: Rankings = {}
    reused_families = 0
    for representation in REPRESENTATIONS:

        def compute(representation=representation):
            indexes = {doc: PreparedBM25(pages, representation) for doc, pages in documents.items()}
            return {
                "rankings": {
                    spec.name: [
                        indexes[query.document_id].rank(query.question, spec.k1, spec.b)
                        for query in queries
                    ]
                    for spec in bm25_feature_grid()
                    if spec.representation == representation
                }
            }

        value, reused = stage(
            store,
            {**identity, "family": representation},
            compute,
            logger=logger,
            name="families",
        )
        rankings.update(value["rankings"])
        reused_families += int(reused)
        logger.emit("research.family.complete", family=representation, reused=reused)

    def combine():
        views = {
            name: rankings[BM25FeatureSpec(rep, 1.2, 0.75).name]
            for name, rep in FUSION_VIEWS.items()
        }
        policies = {}
        for spec in fusion_grid():
            policies[spec.name] = [
                list(
                    fuse_ranks(
                        [views[name][index] for name in spec.systems],
                        spec.weights,
                        method=spec.method,
                        rrf_k=spec.rrf_k,
                    )
                )
                for index in range(len(queries))
            ]
        for policy in exploration_grid():
            policies[policy.name] = [
                list(
                    exploration_order(
                        views["B"][index],
                        [views[name][index] for name in policy.challengers],
                        baseline_keep=policy.baseline_keep,
                    )
                )
                for index in range(len(queries))
            ]
        return {"rankings": policies}

    value, reused = stage(
        store, {**identity, "family": "fusion"}, combine, logger=logger, name="families"
    )
    rankings.update(value["rankings"])
    reused_families += int(reused)
    for orders in rankings.values():
        if len(orders) != len(queries):
            raise ValueError("Incomplete candidate query coverage")
        for query, order in zip(queries, orders, strict=True):
            if sorted(order) != list(range(1, len(documents[query.document_id]) + 1)):
                raise ValueError("Candidate dropped or duplicated a physical page")
    return rankings, {"families": 12, "reused_families": reused_families}


def select_candidate(
    names: Sequence[str],
    rankings: Rankings,
    references: Sequence[ReferenceRecord],
    *,
    conservative: bool,
) -> tuple[str, dict[str, Any]]:
    """Accept only training documents; held-out labels are outside this boundary."""
    if not references or len({row.document_id for row in references}) < 2:
        raise ValueError("Selection needs at least two complete training documents")
    if not names or BASELINE not in rankings:
        raise ValueError("Selection requires candidates and the frozen baseline")
    if any(len(rankings[name]) != len(references) for name in {*names, BASELINE}):
        raise ValueError("Selection ranking/reference coverage differs")
    docs = sorted({reference.document_id for reference in references})

    def scores(name: str) -> list[tuple[float, ...]]:
        values = [
            asdict(evidence_metrics(order, reference.evidence_pages))
            for order, reference in zip(rankings[name], references, strict=True)
        ]
        return [
            tuple(
                fmean(
                    value[metric]
                    for value, ref in zip(values, references)
                    if ref.document_id == doc
                )
                for metric in METRICS
            )
            for doc in docs
        ]

    baseline = scores(BASELINE)
    unique: list[str] = []
    seen = set()
    # Prefer the no-change option in conservative ties, before redundancy screening.
    candidates = list(dict.fromkeys(([BASELINE] if conservative else []) + list(names)))
    for name in candidates:
        signature = tuple(tuple(order) for order in rankings[name])
        if signature not in seen:
            unique.append(name)
            seen.add(signature)
    eligible = []
    for name in unique:
        values = scores(name)
        gains = [value[0] - base[0] for value, base in zip(values, baseline, strict=True)]
        if (
            conservative
            and name != BASELINE
            and (
                min(gains) < -1e-12
                or sum(gain > 1e-12 for gain in gains) < 2
                or fmean(gains) < 0.02
            )
        ):
            continue
        eligible.append((name, tuple(fmean(row[i] for row in values) for i in range(4))))
    selected = max(eligible, key=lambda item: item[1])[0]
    return selected, {
        "training_documents": len(docs),
        "training_questions": len(references),
        "generated": len(candidates),
        "unique_training_rankings": len(unique),
        "duplicates_rejected_on_training_only": len(candidates) - len(unique),
        "eligible_after_gate": len(eligible),
    }


def evaluate_candidates(
    rankings: Rankings, references: Sequence[ReferenceRecord]
) -> dict[str, Any]:
    """Score saved candidates and expose every held-out document and selected policy."""
    aliases = document_aliases(tuple(references))
    families = {
        "free_bm25": [spec.name for spec in bm25_feature_grid()],
        "conservative_fusion": [s.name for s in fusion_grid()]
        + [s.name for s in exploration_grid()],
    }
    folds, policies = [], {}
    for policy, names in families.items():
        held_out_scores = []
        selections = []
        for doc, alias in aliases.items():
            training = [i for i, ref in enumerate(references) if ref.document_id != doc]
            testing = [i for i, ref in enumerate(references) if ref.document_id == doc]
            selected, audit = select_candidate(
                names,
                {name: [rankings[name][i] for i in training] for name in {*names, BASELINE}},
                [references[i] for i in training],
                conservative=policy == "conservative_fusion",
            )
            selections.append(selected)
            values = [
                asdict(evidence_metrics(rankings[selected][i], references[i].evidence_pages))
                for i in testing
            ]
            held_out_scores.extend(values)
            folds.append(
                {
                    "policy": policy,
                    "held_out_document": alias,
                    "held_out_questions": len(testing),
                    "selected_candidate": selected,
                    "selection_audit": audit,
                    "metrics": {metric: fmean(row[metric] for row in values) for metric in METRICS},
                }
            )
        policies[policy] = {
            **{metric: fmean(row[metric] for row in held_out_scores) for metric in METRICS},
            "baseline_selected_folds": selections.count(BASELINE),
            "distinct_selected_candidates": len(set(selections)),
        }
    unique_by_stage = {}
    for name, names in {
        "bm25": families["free_bm25"],
        "fusion": families["conservative_fusion"],
    }.items():
        unique_by_stage[name] = len({tuple(tuple(order) for order in rankings[n]) for n in names})

    def pooled(name: str) -> dict[str, float]:
        values = [
            asdict(evidence_metrics(order, reference.evidence_pages))
            for order, reference in zip(rankings[name], references, strict=True)
        ]
        return {
            metric.replace("_at_k", "_at_5"): fmean(value[metric] for value in values)
            for metric in METRICS
        }

    pooled_scores = {name: pooled(name) for name in families["free_bm25"]}
    family_diagnostics = []
    for representation in REPRESENTATIONS:
        names = [s.name for s in bm25_feature_grid() if s.representation == representation]
        best = max(names, key=lambda name: tuple(pooled_scores[name].values()))
        family_diagnostics.append(
            {
                "representation": representation,
                "candidate": best,
                "promoted": False,
                **pooled_scores[best],
            }
        )
    global_unique = len({tuple(tuple(order) for order in orders) for orders in rankings.values()})
    return {
        "question_count": len(references),
        "document_count": len(aliases),
        "candidate_count": len(rankings),
        "unique_rankings_by_stage": unique_by_stage,
        "globally_unique_rankings": global_unique,
        "baseline": {"candidate": BASELINE, **pooled_scores[BASELINE]},
        "candidate_space": {
            "scalar_bm25_features_generated": len(families["free_bm25"]),
            "fusion_and_exploration_policies_generated": len(families["conservative_fusion"]),
            "total_candidate_configurations_generated": len(rankings),
            "unique_bm25_ranking_signatures": unique_by_stage["bm25"],
            "unique_fusion_ranking_signatures": unique_by_stage["fusion"],
            "duplicate_bm25_rankings_rejected": len(families["free_bm25"])
            - unique_by_stage["bm25"],
            "duplicate_fusion_rankings_rejected": len(families["conservative_fusion"])
            - unique_by_stage["fusion"],
            "globally_unique_ranking_signatures": global_unique,
            "global_duplicate_rankings": len(rankings) - global_unique,
            "cross_stage_duplicates": sum(unique_by_stage.values()) - global_unique,
        },
        "nested_leave_one_document_out": {
            label: {
                **{key.replace("_at_k", "_at_5"): value for key, value in policies[policy].items()},
                "fold_count": len(aliases),
            }
            for policy, label in (
                ("free_bm25", "free_bm25_candidate_selection"),
                ("conservative_fusion", "conservative_fusion_selection_with_baseline_option"),
            )
        },
        "feature_family_diagnostics_pooled_only": family_diagnostics,
        "decision": {
            "changed": False,
            "production_retriever": BASELINE,
            "reason": "The source-bound development audit does not automatically promote a challenger. Inspect all document folds and retain the fixed production contract.",
        },
        "selection_results": policies,
        "document_folds": folds,
        "production_changed": False,
        "ranking_sha256": digest(encode(rankings)),
        "selection_boundary": "Only training-document rankings and labels enter each selector; held-out labels enter scoring afterward. Full-corpus deduplication counts are descriptive only.",
    }
