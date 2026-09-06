"""Inspect, verify or assemble a private LAVA CSV; never upload to Kaggle."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

from lava.evaluation.semantic import encode
from lava.evaluation.submission import (
    build_submission,
    load_submission_inputs,
    sha256,
    validate_submission_csv,
)
from lava.evaluation.submission_store import atomic_write, cached_source, persist_submission
from lava.notebook_support import find_repo_root
from lava.readers.runtime_logging import RuntimeEventLogger


def main() -> int:
    """Keep cloud access explicit and log progress without displaying test content."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preview", "check", "build"), default="preview")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--page-counts", type=Path)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()
    if args.mode == "build" and any(
        value is None for value in (args.predictions, args.page_counts, args.provenance)
    ):
        parser.error("build requires --predictions, --page-counts and --provenance")
    root = find_repo_root(Path(__file__).resolve())
    contract = json.loads((root / "configs/submission.json").read_bytes())
    attempt = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    log_path = root / "artifacts/submission/logs" / f"{attempt}.jsonl"
    logger = RuntimeEventLogger("submission", jsonl_path=log_path)
    started = time.monotonic()
    s3 = None
    bucket = None
    failure = None
    try:
        with logger.stage("submission", heartbeat_seconds=15):
            if args.mode == "preview":
                print(
                    json.dumps(
                        {
                            "competition": contract["competition"],
                            "required_questions": contract["expected_question_count"],
                            "required_documents": contract["expected_document_count"],
                            "required_evidence_scope": "retrieved_full_document_pages",
                            "late_submission_account_eligibility": "unverified",
                            "new_compute": False,
                            "uploaded_to_kaggle": False,
                        },
                        indent=2,
                    )
                )
            else:
                load_dotenv(root / ".env")
                bucket = os.environ.get("S3_BUCKET")
                if not bucket:
                    raise ValueError("S3_BUCKET is required in the environment or project .env")
                s3 = boto3.client(
                    "s3",
                    region_name=os.environ.get("AWS_REGION", "us-west-2"),
                    config=Config(
                        connect_timeout=10,
                        read_timeout=60,
                        retries={"mode": "standard", "total_max_attempts": 3},
                    ),
                )
                payloads = {
                    name: cached_source(
                        s3, bucket, root / "artifacts/submission/inputs" / name, source, logger
                    )
                    for name, source in contract["source_files"].items()
                }
                inputs = load_submission_inputs(
                    payloads["test.csv"], payloads["sample_submission.csv"], contract
                )
                logger.emit("submission.inputs.verified", question_count=len(inputs.order))
                if args.mode == "build":
                    assert args.predictions and args.page_counts and args.provenance
                    predictions = args.predictions.read_bytes()
                    counts = json.loads(args.page_counts.read_bytes())
                    provenance = json.loads(args.provenance.read_bytes())
                    csv_payload = build_submission(inputs, predictions, counts, provenance)
                    check = validate_submission_csv(csv_payload, inputs, counts)
                    manifest = {
                        "schema_version": 1,
                        "competition": contract["competition"],
                        "source_sha256": inputs.source_sha256,
                        "prediction_sha256": sha256(predictions),
                        "provenance": provenance,
                        "submission_sha256": sha256(csv_payload),
                        "validation": check,
                    }
                    target = persist_submission(s3, bucket, root, csv_payload, manifest, logger)
                    print(target)
                else:
                    readiness = {
                        "schema_version": 1,
                        "competition": contract["competition"],
                        "input_status": "verified_against_pinned_s3_sources",
                        "source_sha256": inputs.source_sha256,
                        "test_question_count": len(inputs.order),
                        "test_document_count": contract["expected_document_count"],
                        "answer_format_counts": contract["answer_format_counts"],
                        "language_counts": contract["language_counts"],
                        "submission_created": False,
                        "uploaded_to_kaggle": False,
                        "blockers": [
                            "Complete 624-question test inference with retrieved evidence",
                            "Verified test PDF page counts and prediction provenance",
                            "End-to-end runtime and memory validation against organizer limits",
                            "Authenticated confirmation of late-submission eligibility",
                        ],
                    }
                    atomic_write(root / "reports/submission/readiness.json", encode(readiness))
                    logger.emit("submission.readiness.saved", blockers=len(readiness["blockers"]))
        logger.emit(
            "submission.finished", total_elapsed_seconds=round(time.monotonic() - started, 3)
        )
    except BaseException as error:
        failure = error
        raise
    finally:
        if s3 is not None and bucket is not None:
            try:
                s3.put_object(
                    Bucket=bucket,
                    Key=f"experiments/submissions/logs/{attempt}.jsonl",
                    Body=log_path.read_bytes(),
                    ContentType="application/x-ndjson",
                    IfNoneMatch="*",
                )
            except Exception:
                logger.emit("submission.log.persistence_failed", level="ERROR")
                if failure is None:
                    raise
    print(
        "SUBMISSION_INPUTS_VERIFIED" if args.mode == "check" else "SUBMISSION_PREPARATION_COMPLETE"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
