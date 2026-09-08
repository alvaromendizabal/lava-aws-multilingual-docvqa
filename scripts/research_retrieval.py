"""Reproduce the full lexical catalog and document-isolated selection on CPU."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import boto3
from botocore.config import Config
from dotenv import load_dotenv

from lava.evaluation.semantic import digest, encode
from lava.evaluation.submission_store import atomic_write, cached_source
from lava.notebook_support import find_repo_root
from lava.readers.oracle_assets import document_aliases, parse_training_csv
from lava.readers.runtime_logging import RuntimeEventLogger
from lava.retrieval.evaluation import pilot_sources
from lava.retrieval.lexical import PageText, RetrievalQuery
from lava.retrieval.pipeline import extract_pages, load_public_report, stage
from lava.retrieval.research import build_rankings, evaluate_candidates, research_contract
from lava.retrieval.storage import CheckpointStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preview", "evaluate"), default="preview")
    args = parser.parse_args()
    root = find_repo_root()
    baseline = load_public_report(root)
    contract = research_contract(root, baseline["contract_id"])
    identity = digest(encode(contract))
    attempt = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    log = root / "artifacts/retrieval/research" / f"{attempt}.jsonl"
    logger = RuntimeEventLogger("retrieval.research", jsonl_path=log)
    logger.emit("research.plan", candidates=1582, contract_id=identity, new_compute=False)
    if args.mode == "preview":
        return 0
    load_dotenv(root / ".env", override=False)
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        raise ValueError("S3_BUCKET is required in the existing project environment")
    s3 = boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION", "us-west-2"),
        config=Config(connect_timeout=10, read_timeout=60, retries={"mode": "standard"}),
    )
    store = CheckpointStore(s3, bucket, f"experiments/submissions/research/{identity}")
    retrieval = CheckpointStore(
        s3, bucket, f"experiments/submissions/retrieval/{baseline['contract_id']}"
    )
    started = time.perf_counter()
    succeeded = False
    try:
        with logger.stage("research.run", heartbeat_seconds=15):
            sources = pilot_sources(root, baseline["implementation"]["config"])
            cache = root / "artifacts/retrieval/inputs"
            references = tuple(
                sorted(
                    parse_training_csv(
                        cached_source(s3, bucket, cache / "train.csv", sources["train.csv"], logger)
                    ),
                    key=lambda ref: ref.question_id,
                )
            )
            if len(references) != 16 or len(document_aliases(references)) != 5:
                raise ValueError("The research protocol requires the complete 16-question pilot")
            documents = {}
            for document in document_aliases(references):
                source = sources[f"{document}.pdf"]

                def extract(source=source, document=document):
                    return extract_pages(
                        cached_source(s3, bucket, cache / f"{document}.pdf", source, logger)
                    )

                pages, _ = stage(
                    retrieval,
                    {"implementation": baseline["contract_id"], "source": source},
                    extract,
                    logger=logger,
                    name="documents",
                )
                if pages["pdf_sha256"] != source["sha256"]:
                    raise ValueError("PDF extraction source mismatch")
                documents[document] = tuple(PageText(**page) for page in pages["pages"])
            queries = tuple(
                RetrievalQuery(ref.question_id, ref.document_id, ref.question) for ref in references
            )
            rankings, recovery = build_rankings(queries, documents, store, contract, logger)
            # This checkpoint contains no answer or evidence labels and is read back first.
            ranked = {"queries": [asdict(query) for query in queries], "rankings": rankings}
            store.write("rankings.json", ranked)
            if store.read("rankings.json") != ranked:
                raise ValueError("Rankings failed independent durable read-back")
            with logger.stage("research.document_selection", heartbeat_seconds=15):
                summary = {
                    "schema_version": 1,
                    "status": "Executable feature audit complete",
                    "contract": contract,
                    "contract_id": identity,
                    **evaluate_candidates(rankings, references),
                }
            store.write("summary.json", summary)
            if store.read("summary.json") != summary:
                raise ValueError("Research summary failed durable read-back")
            path = root / "reports/retrieval/feature_search.json"
            payload = encode(summary)
            atomic_write(path, payload)
            atomic_write(path.with_suffix(".sha256"), (digest(payload) + "\n").encode())
            receipt: dict[str, Any] = {
                "attempt": attempt,
                "contract_id": identity,
                "summary_sha256": digest(payload),
                "elapsed_seconds": time.perf_counter() - started,
                **recovery,
            }
            store.write(f"attempts/{attempt}.json", receipt)
            if store.read(f"attempts/{attempt}.json") != receipt:
                raise ValueError("Research execution receipt failed read-back")
            atomic_write(root / "artifacts/retrieval/research" / f"{attempt}.json", encode(receipt))
            logger.emit("research.verified", **receipt)
            succeeded = True
    finally:
        try:
            events = {"events": [json.loads(line) for line in log.read_text().splitlines()]}
            store.write(f"logs/{attempt}.json", events)
            if store.read(f"logs/{attempt}.json") != events:
                raise ValueError("Research log failed durable read-back")
        except Exception:
            logger.emit("research.log.persistence_failed", level="ERROR")
            if succeeded:
                raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
