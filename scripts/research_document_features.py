"""Run document-domain ablations on pinned training PDFs, with reusable checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pymupdf

from lava.evaluation.semantic import ImmutableS3Objects, digest, encode
from lava.evaluation.submission_store import atomic_write
from lava.notebook_support import find_repo_root, git_snapshot
from lava.readers.oracle_assets import document_aliases, parse_training_csv
from lava.readers.runtime_logging import RuntimeEventLogger
from lava.retrieval.domain_features import (
    COMPONENTS,
    POLICY,
    DocumentPage,
    evaluate_ablation,
    extract_document,
    page_features,
)
from lava.retrieval.evaluation import pilot_sources
from lava.retrieval.lexical import RetrievalQuery
from lava.retrieval.pipeline import ObjectStore, stage


class LocalArchive:
    """Verified local staging; separately archive it remotely for durable operation."""

    def __init__(self, path: Path):
        self.path = path

    def read(self, suffix: str) -> dict[str, Any] | None:
        path = self.path / suffix
        if not path.exists():
            return None
        saved = json.loads(path.read_bytes())
        if saved.get("state") != "complete" or saved.get("sha256") != digest(
            encode(saved.get("value"))
        ):
            raise ValueError("Local research checkpoint is incomplete or corrupt")
        return saved["value"]

    def write(self, suffix: str, value: dict[str, Any]) -> None:
        existing = self.read(suffix)
        if existing is not None:
            if existing != value:
                raise ValueError("Conflicting immutable local research checkpoint")
            return
        atomic_write(
            self.path / suffix,
            encode({"state": "complete", "sha256": digest(encode(value)), "value": value}),
        )


def contract_for(root: Path) -> dict[str, Any]:
    files = (
        "src/lava/retrieval/domain_features.py",
        "scripts/research_document_features.py",
        "src/lava/retrieval/lexical.py",
        "src/lava/retrieval/research.py",
        "src/lava/retrieval/feature_research.py",
        "src/lava/retrieval/pipeline.py",
        "uv.lock",
    )
    config = json.loads((root / "configs/retrieval.json").read_bytes())
    return {
        "schema_version": 1,
        "policy": POLICY,
        "components": list(COMPONENTS),
        "sources": pilot_sources(root, config),
        "pymupdf_version": pymupdf.VersionBind,
        "source_sha256": {p: digest((root / p).read_bytes()) for p in files},
        "scope": "Training diagnostic only; fixed catalog; no test labels; no automatic promotion",
    }


def run(
    root: Path,
    inputs: Path,
    store: ObjectStore,
    contract: dict[str, Any],
    logger: RuntimeEventLogger,
) -> tuple[dict, dict]:
    identity = digest(encode(contract))
    store.write("contract.json", contract)
    payloads = {}
    for name, source in contract["sources"].items():
        payload = (inputs / name).read_bytes()
        if digest(payload) != source["sha256"]:
            raise ValueError(f"Pinned training input checksum differs: {name}")
        payloads[name] = payload
    refs = tuple(sorted(parse_training_csv(payloads["train.csv"]), key=lambda r: r.question_id))
    aliases = document_aliases(refs)
    if len(refs) != 16 or len(aliases) != 5:
        raise ValueError("The complete pinned training diagnostic is required")
    documents, coverage = {}, []
    reused_documents, reused_queries = 0, 0
    for doc, alias in aliases.items():
        value, reused = stage(
            store,
            {"contract_id": identity, "source": contract["sources"][doc + ".pdf"]},
            lambda doc=doc: {
                "pages": [asdict(p) for p in extract_document(payloads[doc + ".pdf"])]
            },
            logger=logger,
            name="documents",
        )
        documents[doc] = tuple(
            DocumentPage(**{**p, "blocks": tuple(p["blocks"])}) for p in value["pages"]
        )
        reused_documents += int(reused)
        coverage.append(
            {
                "document": alias,
                "physical_pages": len(documents[doc]),
                "native_textless_pages": sum(not p.native for p in documents[doc]),
                "table_detected_pages": sum(p.table_count > 0 for p in documents[doc]),
                "table_detection_errors": sum(p.table_error is not None for p in documents[doc]),
                "text_strategy_pages": sum(
                    p.table_strategy == "text" and p.table_count > 0 for p in documents[doc]
                ),
            }
        )
    rankings: dict[str, list[list[int]]] = {}
    active = {name: 0 for name in COMPONENTS}
    for ref in refs:
        query = RetrievalQuery(ref.question_id, ref.document_id, ref.question)
        value, reused = stage(
            store,
            {"contract_id": identity, "query": asdict(query)},
            lambda query=query: page_features(query, documents[query.document_id]),
            logger=logger,
            name="queries",
        )
        reused_queries += int(reused)
        for name, order in value["rankings"].items():
            rankings.setdefault(name, []).append(order)
        for name, enabled in value["active_features"].items():
            active[name] += int(enabled)
    # No reference labels are passed into extraction or query-page scoring.
    # Persist and independently reread every full ranking before gold scoring.
    ranked = {"query_ids": [r.question_id for r in refs], "rankings": rankings}
    store.write("rankings.json", ranked)
    if store.read("rankings.json") != ranked:
        raise ValueError("Ranking read-back failed before reference scoring")
    with logger.stage("domain.document_selection", heartbeat_seconds=15):
        summary = {
            "schema_version": 1,
            "contract_id": identity,
            "contract": contract,
            "ranking_sha256": digest(encode(ranked)),
            "page_coverage": coverage,
            "feature_active_question_counts": active,
            **evaluate_ablation(rankings, refs),
        }
    store.write("summary.json", summary)
    if store.read("summary.json") != summary:
        raise ValueError("Summary read-back mismatch")
    return summary, {"reused_documents": reused_documents, "reused_queries": reused_queries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preview", "evaluate"), default="preview")
    parser.add_argument("--inputs", type=Path, default=Path("artifacts/retrieval/inputs"))
    parser.add_argument(
        "--local-archive", type=Path, help="Local staging; remote archival remains required"
    )
    args = parser.parse_args()
    root = find_repo_root()
    contract = contract_for(root)
    identity = digest(encode(contract))
    attempt = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    log_path = root / "artifacts/retrieval/domain" / f"{attempt}.jsonl"
    logger = RuntimeEventLogger("retrieval.domain", jsonl_path=log_path)
    logger.emit("domain.plan", contract_id=identity, candidates=17, new_compute=False)
    if args.mode == "preview":
        return 0
    store: ObjectStore
    if args.local_archive:
        store = LocalArchive(args.local_archive / identity)
    else:
        import boto3

        bucket = os.environ["S3_BUCKET"]
        client = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-2"))
        store = ImmutableS3Objects(
            client, bucket, f"experiments/submissions/domain-research/{identity}"
        )
    started = time.perf_counter()
    try:
        with logger.stage("domain.run", heartbeat_seconds=15):
            summary, reused = run(root, args.inputs, store, contract, logger)
            receipt = {
                "attempt": attempt,
                "contract_id": identity,
                "summary_sha256": digest(encode(summary)),
                "elapsed_seconds": time.perf_counter() - started,
                "source": git_snapshot(root),
                "backend": "local_staging_requires_remote_archive" if args.local_archive else "s3",
                "new_compute": False,
                "model_calls": 0,
                **reused,
            }
            store.write(f"attempts/{attempt}.json", receipt)
            if store.read(f"attempts/{attempt}.json") != receipt:
                raise ValueError("Execution receipt read-back mismatch")
            path = root / "reports/retrieval/document_features.json"
            atomic_write(path, encode(summary))
            atomic_write(path.with_suffix(".sha256"), (digest(encode(summary)) + "\n").encode())
            logger.emit("domain.verified", **receipt)
    finally:
        events = {"events": [json.loads(line) for line in log_path.read_text().splitlines()]}
        store.write(f"logs/{attempt}.json", events)
        if store.read(f"logs/{attempt}.json") != events:
            raise ValueError("Execution log read-back mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
