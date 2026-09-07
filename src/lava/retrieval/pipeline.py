"""Checksum-bound extraction and ranking stages with durable, independently reusable outputs."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

import pymupdf

from lava.evaluation.semantic import encode
from lava.evaluation.submission_store import atomic_write
from lava.readers.runtime_logging import RuntimeEventLogger
from lava.retrieval.lexical import BM25Index, PageText, RetrievalQuery


class ObjectStore(Protocol):
    """Minimal immutable JSON persistence boundary, implemented by the S3 store."""

    def read(self, suffix: str) -> dict[str, Any] | None: ...

    def write(self, suffix: str, value: dict[str, Any]) -> None: ...


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stage(
    store: ObjectStore,
    identity: dict[str, Any],
    compute: Callable[[], dict[str, Any]],
    *,
    logger: RuntimeEventLogger,
    name: str,
) -> tuple[dict[str, Any], bool]:
    """Reuse verified work; acknowledge progress only after durable read-back."""
    key = f"{name}/{digest(encode(identity))}.json"
    saved = store.read(key)
    if saved is not None:
        if saved.get("identity") != identity or digest(encode(saved.get("result"))) != saved.get(
            "result_sha256"
        ):
            raise ValueError("Retrieval checkpoint identity or checksum mismatch")
        logger.emit("retrieval.stage.reused", stage=name, checkpoint_id=digest(encode(identity)))
        return saved["result"], True
    with logger.stage(f"retrieval.{name}", heartbeat_seconds=15):
        started = time.perf_counter()
        result = compute()
        envelope = {
            "identity": identity,
            "result": result,
            "result_sha256": digest(encode(result)),
        }
        store.write(key, envelope)
        if store.read(key) != envelope:
            raise ValueError("Retrieval checkpoint read-back mismatch")
        logger.emit(
            "retrieval.stage.persisted",
            stage=name,
            checkpoint_id=digest(encode(identity)),
            compute_and_persist_seconds=time.perf_counter() - started,
        )
    return result, False


def extract_pages(payload: bytes) -> dict[str, Any]:
    """Read every physical page; empty/error pages stay visible in the candidate set."""
    pages: list[dict[str, Any]] = []
    with pymupdf.open(stream=payload, filetype="pdf") as document:
        if document.needs_pass or not len(document):
            raise ValueError("PDF must be readable and nonempty")
        for index in range(len(document)):
            try:
                text = document[index].get_text("text", sort=True).strip()
            except (RuntimeError, ValueError):
                # Retain the failed physical page, never shorten the denominator.
                pages.append(asdict(PageText(index + 1, "", "extraction_error")))
            else:
                pages.append(asdict(PageText(index + 1, text, "ok" if text else "textless")))
    return {"pdf_sha256": digest(payload), "pages": pages}


def rank_query(query: RetrievalQuery, pages: tuple[PageText, ...], config: dict[str, Any]) -> dict:
    """Rank all pages and retain the query-independent page-order control."""
    ranked = BM25Index(pages, k1=config["k1"], b=config["b"]).rank(query.question)
    return {
        "question_id": query.question_id,
        "document_id": query.document_id,
        "bm25": [number for number, _ in ranked],
        "scores": [score for _, score in ranked],
        "page_order": [page.number for page in pages],
        "zero_signal": all(score == 0 for _, score in ranked),
    }


def implementation_contract(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Pin the actual implementation and dependencies without changing the reader judge."""
    paths = [
        *[
            root / "src/lava/retrieval" / name
            for name in ("lexical.py", "pipeline.py", "storage.py", "evaluation.py")
        ],
        root / "src/lava/evaluation/retrieval.py",
        root / "src/lava/readers/oracle_assets.py",
        root / "scripts/evaluate_retrieval.py",
        root / "uv.lock",
    ]
    return {
        "schema_version": 1,
        "config": config,
        "pymupdf_version": pymupdf.VersionBind,
        "source_sha256": {str(path.relative_to(root)): digest(path.read_bytes()) for path in paths},
    }


def save_public_report(root: Path, summary: dict[str, Any]) -> None:
    """Materialize the public aggregate and checksum together from durable results."""
    payload = encode(summary)
    path = root / "reports/retrieval/summary.json"
    atomic_write(path, payload)
    atomic_write(path.with_suffix(".sha256"), (digest(payload) + "\n").encode())


def load_public_report(root: Path) -> dict[str, Any]:
    path = root / "reports/retrieval/summary.json"
    payload = path.read_bytes()
    if digest(payload) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Public retrieval report checksum mismatch")
    summary = json.loads(payload)
    config = json.loads((root / "configs/retrieval.json").read_bytes())
    if summary["implementation"] != implementation_contract(root, config):
        raise ValueError("Retrieval report is stale for the current implementation")
    return summary
