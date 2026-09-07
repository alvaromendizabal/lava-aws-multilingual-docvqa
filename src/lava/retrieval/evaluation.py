"""Full-document retrieval diagnostics on the frozen labeled pilot."""

from __future__ import annotations

import csv
import io
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from statistics import fmean
from typing import Any

from lava.evaluation.retrieval import aggregate_retrieval_scores, score_retrieval_budgets
from lava.evaluation.schemas import ReferenceRecord
from lava.evaluation.semantic import encode
from lava.evaluation.submission_store import cached_source
from lava.notebook_support import git_snapshot
from lava.readers.oracle_assets import document_aliases, parse_training_csv
from lava.readers.runtime_logging import RuntimeEventLogger
from lava.retrieval.lexical import PageText, RetrievalQuery
from lava.retrieval.pipeline import (
    digest,
    extract_pages,
    implementation_contract,
    rank_query,
    save_public_report,
    stage,
)
from lava.retrieval.storage import CheckpointStore


def pilot_sources(root: Path, config: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Accept only the previously verified manifest and its complete training partition."""
    payload = (root / "reports/raw_data_manifest.csv").read_bytes()
    if digest(payload) != config["source_manifest_sha256"]:
        raise ValueError("Raw source manifest checksum changed")
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    selected = [
        row for row in rows if row["name"] == "train.csv" or row["name"].startswith("train_pdfs/")
    ]
    result = {}
    for row in selected:
        name = Path(row["name"]).name
        if name in result:
            raise ValueError("Duplicate training source filename")
        if not row["version_id"] or len(row["sha256"]) != 64:
            raise ValueError("Training sources require a version and SHA-256")
        result[name] = {
            "key": row["s3_key"],
            "version_id": row["version_id"],
            "sha256": row["sha256"],
        }
    if "train.csv" not in result or len(result) != config["expected_document_count"] + 1:
        raise ValueError("Incomplete training source inventory")
    return result


def evaluate_rankings(
    references: tuple[ReferenceRecord, ...],
    rankings: dict[str, dict[str, Any]],
    documents: dict[str, tuple[PageText, ...]],
    *,
    budgets: tuple[int, ...],
) -> dict[str, Any]:
    """Introduce gold evidence only after complete label-blind rankings exist."""
    if not references or len({row.question_id for row in references}) != len(references):
        raise ValueError("References must be nonempty with unique question IDs")
    if set(rankings) != {row.question_id for row in references}:
        raise ValueError("Ranking coverage differs from reference coverage")
    aliases = document_aliases(references)
    if set(documents) != set(aliases):
        raise ValueError("Document coverage differs from reference coverage")
    details: list[dict[str, Any]] = []
    for index, reference in enumerate(sorted(references, key=lambda row: row.question_id), 1):
        pages = documents[reference.document_id]
        ranking = rankings[reference.question_id]
        if (
            ranking["document_id"] != reference.document_id
            or ranking["question_id"] != reference.question_id
        ):
            raise ValueError("A ranking belongs to a different question or PDF")
        page_set = set(range(1, len(pages) + 1))
        if not set(reference.evidence_pages).issubset(page_set):
            raise ValueError("Gold evidence exceeds the physical PDF page count")
        for method in ("page_order", "bm25"):
            order = tuple(ranking[method])
            if set(order) != page_set or len(order) != len(page_set):
                raise ValueError("Rankings must include every physical page exactly once")
            for score in score_retrieval_budgets(
                order, frozenset(reference.evidence_pages), budgets=budgets
            ):
                details.append(
                    {
                        "question": f"q-{index:02d}",
                        "document": aliases[reference.document_id],
                        "language": reference.language,
                        "answer_format": reference.answer_format.value,
                        "method": method,
                        **asdict(score),
                    }
                )
    aggregate: dict[str, Any] = {}
    for method in ("page_order", "bm25"):
        selected = [row for row in details if row["method"] == method]
        grouped = {}
        for dimension in ("document", "language", "answer_format"):
            grouped[dimension] = {
                group: _aggregate([row for row in selected if row[dimension] == group])
                for group in sorted({row[dimension] for row in selected})
            }
        macro = {
            str(k): {
                metric: fmean(values[str(k)][metric] for values in grouped["document"].values())
                for metric in grouped["document"][next(iter(grouped["document"]))][str(k)]
            }
            for k in budgets
        }
        aggregate[method] = {
            "question_average": _aggregate(selected),
            "document_average": macro,
            "by": grouped,
        }
    coverage = []
    for document_id, pages in sorted(documents.items()):
        gold = {
            page
            for row in references
            if row.document_id == document_id
            for page in row.evidence_pages
        }
        coverage.append(
            {
                "document": aliases[document_id],
                "page_count": len(pages),
                "textless_pages": sum(page.status == "textless" for page in pages),
                "extraction_errors": sum(page.status == "extraction_error" for page in pages),
                "native_text_characters": sum(len(page.text) for page in pages),
                "unique_gold_pages": len(gold),
                "gold_pages_without_text": sum(not pages[number - 1].text for number in gold),
            }
        )
    return {
        "methods": aggregate,
        "per_question": details,
        "document_coverage": coverage,
        "zero_signal_questions": sum(row["zero_signal"] for row in rankings.values()),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from lava.evaluation.retrieval import RetrievalScore

    fields = (
        "k",
        "recall_at_k",
        "all_evidence_at_k",
        "reciprocal_rank_at_k",
        "average_precision_at_k",
        "ndcg_at_k",
    )
    scores = tuple(RetrievalScore(**{key: row[key] for key in fields}) for row in rows)
    return {str(k): values for k, values in aggregate_retrieval_scores(scores).items()}


def run_evaluation(
    root: Path, s3: Any, bucket: str, config: dict[str, Any], logger: RuntimeEventLogger
) -> dict[str, Any]:
    """Reuse PDF/query checkpoints across interruption; publish only complete aggregates."""
    started = time.perf_counter()
    sources = pilot_sources(root, config)
    implementation = implementation_contract(root, config)
    identity = digest(encode(implementation))
    store = CheckpointStore(s3, bucket, f"experiments/submissions/retrieval/{identity}")
    cache = root / "artifacts/retrieval/inputs"
    references = parse_training_csv(
        cached_source(s3, bucket, cache / "train.csv", sources["train.csv"], logger)
    )
    if len(references) != config["expected_question_count"]:
        raise ValueError("Unexpected labeled question count")
    aliases = document_aliases(references)
    if {f"{name}.pdf" for name in aliases} != set(sources) - {"train.csv"}:
        raise ValueError("Training questions and PDF inventory differ")
    documents = {}
    reused_documents = 0
    for document_id, alias in aliases.items():
        source = sources[f"{document_id}.pdf"]

        def extract(source=source, document_id=document_id):
            payload = cached_source(s3, bucket, cache / f"{document_id}.pdf", source, logger)
            return extract_pages(payload)

        result, reused = stage(
            store,
            {"implementation": identity, "source": source},
            extract,
            logger=logger,
            name="documents",
        )
        if result["pdf_sha256"] != source["sha256"]:
            raise ValueError("Extracted document does not match its source")
        pages = tuple(PageText(**page) for page in result["pages"])
        documents[document_id] = pages
        reused_documents += int(reused)
        logger.emit(
            "retrieval.document.completed", document=alias, page_count=len(pages), reused=reused
        )
    rankings = {}
    reused_queries = 0
    for number, reference in enumerate(references, 1):
        query = RetrievalQuery(reference.question_id, reference.document_id, reference.question)
        pages = documents[query.document_id]
        result, reused = stage(
            store,
            {
                "implementation": identity,
                "query": asdict(query),
                "pages_sha256": digest(encode([asdict(page) for page in pages])),
            },
            partial(rank_query, query, pages, config["bm25"]),
            logger=logger,
            name="queries",
        )
        rankings[query.question_id] = result
        reused_queries += int(reused)
        logger.emit(
            "retrieval.question.completed", completed=number, total=len(references), reused=reused
        )
    metrics = evaluate_rankings(references, rankings, documents, budgets=tuple(config["budgets"]))
    existing = store.read("summary.json")
    if existing is None:
        summary = {
            "schema_version": 1,
            "status": "Full-document retrieval evaluated",
            "split": "training_diagnostic",
            "implementation": implementation,
            "contract_id": identity,
            "code_commit": git_snapshot(root)["git_commit_sha"],
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "first_run_seconds": round(time.perf_counter() - started, 3),
            "question_count": len(references),
            "document_count": len(documents),
            "language_counts": dict(Counter(row.language for row in references)),
            "answer_format_counts": dict(Counter(row.answer_format.value for row in references)),
            "reader_evaluated": False,
            "local_lava_overall": None,
            **metrics,
        }
        store.write("summary.json", summary)
        if store.read("summary.json") != summary:
            raise ValueError("Retrieval summary failed durable read-back verification")
    else:
        summary = existing
        if summary["implementation"] != implementation or any(
            summary[key] != value for key, value in metrics.items()
        ):
            raise ValueError("Completed retrieval summary differs from verified checkpoints")
    save_public_report(root, summary)
    attempt: dict[str, Any] = {
        "contract_id": identity,
        "documents_reused": reused_documents,
        "documents_computed": len(documents) - reused_documents,
        "queries_reused": reused_queries,
        "queries_computed": len(references) - reused_queries,
        "total_elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    logger.emit("retrieval.evaluation.completed", **attempt)
    return attempt
