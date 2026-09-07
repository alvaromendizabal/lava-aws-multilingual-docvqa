"""Preview or run the frozen full-document retrieval pilot on the current CPU host."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import UTC, datetime

import boto3
from botocore.config import Config
from dotenv import load_dotenv

from lava.evaluation.submission_store import atomic_write
from lava.notebook_support import find_repo_root
from lava.readers.runtime_logging import RuntimeEventLogger
from lava.retrieval.evaluation import pilot_sources, run_evaluation
from lava.retrieval.pipeline import load_public_report
from lava.retrieval.reporting import render_report
from lava.retrieval.storage import CheckpointStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preview", "evaluate"), default="preview")
    args = parser.parse_args()
    root = find_repo_root()
    config = json.loads((root / "configs/retrieval.json").read_bytes())
    attempt = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    log = root / "artifacts/retrieval/logs" / f"{attempt}.jsonl"
    logger = RuntimeEventLogger("retrieval", jsonl_path=log)
    sources = pilot_sources(root, config)
    if args.mode == "preview":
        logger.emit(
            "retrieval.preview",
            questions=config["expected_question_count"],
            documents=len(sources) - 1,
            methods=["page_order", "bm25"],
            budgets=config["budgets"],
            creates_compute=False,
            reads_private_sources=False,
        )
        return 0
    load_dotenv(root / ".env", override=False)
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        raise ValueError("S3_BUCKET is required in the project environment")
    s3 = boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION", "us-west-2"),
        config=Config(
            connect_timeout=10,
            read_timeout=60,
            retries={"mode": "standard", "total_max_attempts": 3},
        ),
    )
    failure = None
    try:
        with logger.stage("retrieval.run", heartbeat_seconds=15):
            run_evaluation(root, s3, bucket, config, logger)
            atomic_write(
                root / "reports/retrieval/index.html",
                render_report(load_public_report(root)).encode(),
            )
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            store = CheckpointStore(s3, bucket, "experiments/submissions/retrieval/logs")
            value = {
                "attempt": attempt,
                "events": [json.loads(line) for line in log.read_text().splitlines()],
            }
            store.write(f"{attempt}.json", value)
            if store.read(f"{attempt}.json") != value:
                raise ValueError("Retrieval log read-back mismatch")
            logger.emit("retrieval.log.archived", attempt=attempt)
        except Exception:
            logger.emit("retrieval.log.persistence_failed", level="ERROR")
            if failure is None:
                raise
    print("RETRIEVAL_EVALUATION_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
